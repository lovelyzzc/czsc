"""V2 横截面因子、缓冲选股和择时对照的冻结决策内核。

V1 是不可修改的审计基线；本模块只实现 V2 的决策层和非对称判定树。正式前向运行必须
先通过 :mod:`xs_chan_data_v2` 的内容寻址数据门，并由 :mod:`xs_chan_oos_ledger` 记录。
本模块不会把历史回填结果升级为确认性证据。

常用命令::

    uv run --no-sync python scripts/xs_chan_research_v2.py protocol-check
    uv run --no-sync python scripts/xs_chan_research_v2.py plan \
        --features decision_features.parquet --states chan_states.parquet \
        --schedule weekly_schedule.parquet --all-adv all_decision_adv.parquet \
        --output-dir /tmp/xs_chan_v2_plan
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROTOCOL_PATH = SCRIPT_DIR / "xs_chan_protocol_v2.json"
STATE_COLUMNS = ("symbol", "dt", "regime")
RESEARCH_LABEL = "CONTAMINATED_STRESS_ONLY"
RNG_ALGORITHM_VERSION = "xs_chan_rng_v2_sha256_seedsequence_v1"
CONFIRMATORY_COMPARISONS = (
    "F_2x_minus_R_match_2x",
    "FC_gross_minus_FGR_gross",
    "FC_2x_minus_FGR_2x",
    "FC_gross_minus_FMA_gross",
    "FC_2x_minus_FMA_2x",
    "FMA_gross_minus_FMGR_gross",
    "FMA_2x_minus_FMGR_2x",
)
CORPORATE_ACTION_TYPES = (
    "cash_dividend",
    "share_change",
    "rights_issue",
    "delist_cash",
    "delist_share",
    "delist_writeoff",
)
UNIVERSE_RECONCILIATION_PARTITIONS = (
    "security_master:list_status=L",
    "security_master:list_status=D",
    "security_master:list_status=P",
    "raw_daily:bundle_symbol_set",
    "corporate_actions:coverage_universe",
)
REQUIRED_ENGINEERING_GATES = (
    "state_recompute_engineering",
    "feature_prefix_engineering",
    "source_archive_reconciliation_engineering",
    "suspension_reconciliation_engineering",
    "calendar_ledger_binding_engineering",
    "execution_replay_engineering",
    "corporate_action_replay_engineering",
    "statistics_replay_engineering",
    "artifact_semantic_verifier_engineering",
    "external_timestamp_anchor_engineering",
)
FORWARD_DEPENDENCY_KEYS = (
    "data_evidence",
    "state_engine",
    "feature_engine",
    "research_engine",
    "execution_engine",
    "statistics_engine",
    "replay_verifier",
)


class ProtocolError(ValueError):
    """机器协议与冻结规格不一致。"""


class ControlMatchError(RuntimeError):
    """对照无法精确匹配；正式模式必须使该期失效。"""


@dataclass(frozen=True)
class SelectionSpec:
    """唯一冻结的 V2 选股规格。"""

    target_size: int
    retention_max_rank: int
    ma_window: int


@dataclass(frozen=True)
class EligibilitySpec:
    """PIT 股票池约束；``eligible`` 只能由这些字段推导。"""

    min_market_sessions_since_listing: int
    min_valid_sessions_60: int
    min_adv20_thousand_cny: float
    excluded_boards: tuple[str, ...]


@dataclass(frozen=True)
class V2Config:
    """从机器协议提取的决策参数。"""

    protocol_id: str
    protocol_status: str
    selection: SelectionSpec
    eligibility: EligibilitySpec
    allowed_chan_regimes: tuple[int, ...]
    random_seed: int
    random_seed_count: int
    winsor_low: float
    winsor_high: float
    neutralization_buckets: int
    minimum_complete_weeks: int
    minimum_hac_t: float
    maximum_top10_share: float
    maximum_exposure_gap: float
    bootstrap_method: str
    bootstrap_block_weeks: int
    bootstrap_draws: int
    bootstrap_seed: int
    bootstrap_lower_quantile: float


@dataclass(frozen=True)
class SelectionResult:
    """一次带缓冲的因子名单选择。"""

    symbols: tuple[str, ...]
    reason_by_symbol: Mapping[str, str]

    def __post_init__(self) -> None:
        symbols = tuple(map(str, self.symbols))
        if len(symbols) != len(set(symbols)) or set(symbols) != set(map(str, self.reason_by_symbol)):
            raise ValueError("selection result must contain unique symbols and exact reasons")
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "reason_by_symbol",
            MappingProxyType({str(symbol): str(reason) for symbol, reason in self.reason_by_symbol.items()}),
        )


@dataclass(frozen=True)
class DecisionPlan:
    """决策层产物；不包含成交价格或未来标签。"""

    decision_dt: pd.Timestamp
    exec_dt: pd.Timestamp
    symbols: tuple[str, ...]
    rank_by_symbol: Mapping[str, int]
    selection_reason: Mapping[str, str]
    adv_cny: Mapping[str, float]
    chan_allowed: Mapping[str, bool]
    ma_allowed: Mapping[str, bool]
    stratum_by_symbol: Mapping[str, tuple[str, int, int]]

    def __post_init__(self) -> None:
        if any(not isinstance(symbol, str) or not symbol for symbol in self.symbols):
            raise ValueError("decision-plan symbols must be non-empty strings")
        symbols = tuple(self.symbols)
        symbol_set = set(symbols)
        if len(symbols) != len(symbol_set):
            raise ValueError("decision-plan symbols must be unique")
        if pd.Timestamp(self.exec_dt) <= pd.Timestamp(self.decision_dt):
            raise ValueError("decision-plan exec_dt must be after decision_dt")
        exact_mappings = {
            "rank_by_symbol": self.rank_by_symbol,
            "selection_reason": self.selection_reason,
            "chan_allowed": self.chan_allowed,
            "ma_allowed": self.ma_allowed,
            "stratum_by_symbol": self.stratum_by_symbol,
        }
        for name, mapping in exact_mappings.items():
            if any(not isinstance(key, str) for key in mapping) or set(mapping) != symbol_set:
                raise ValueError(f"{name} keys must exactly match decision-plan symbols")
        if any(not isinstance(key, str) for key in self.adv_cny) or not symbol_set.issubset(set(self.adv_cny)):
            raise ValueError("adv_cny must cover every decision-plan symbol")
        if any(
            isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer)) or int(rank) <= 0
            for rank in self.rank_by_symbol.values()
        ):
            raise ValueError("factor ranks must be positive integers without coercion")
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not math.isfinite(float(value))
            or float(value) <= 0
            for value in self.adv_cny.values()
        ):
            raise ValueError("adv_cny values must be finite and positive")
        if any(not isinstance(value, str) or not value for value in self.selection_reason.values()):
            raise ValueError("selection reasons must be non-empty strings")
        for name, mapping in (("chan_allowed", self.chan_allowed), ("ma_allowed", self.ma_allowed)):
            if any(not isinstance(value, (bool, np.bool_)) for value in mapping.values()):
                raise ValueError(f"{name} values must be literal booleans")
        for value in self.stratum_by_symbol.values():
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or not isinstance(value[0], str)
                or not value[0]
                or any(
                    isinstance(bucket, (bool, np.bool_)) or not isinstance(bucket, (int, np.integer))
                    for bucket in value[1:]
                )
            ):
                raise ValueError("strata must be non-empty industry plus exact integer mcap/ADV buckets")
        object.__setattr__(self, "decision_dt", pd.Timestamp(self.decision_dt))
        object.__setattr__(self, "exec_dt", pd.Timestamp(self.exec_dt))
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "rank_by_symbol",
            MappingProxyType({symbol: int(value) for symbol, value in self.rank_by_symbol.items()}),
        )
        object.__setattr__(
            self,
            "selection_reason",
            MappingProxyType(dict(self.selection_reason)),
        )
        object.__setattr__(
            self,
            "adv_cny",
            MappingProxyType({symbol: float(value) for symbol, value in self.adv_cny.items()}),
        )
        object.__setattr__(
            self,
            "chan_allowed",
            MappingProxyType({symbol: bool(value) for symbol, value in self.chan_allowed.items()}),
        )
        object.__setattr__(
            self,
            "ma_allowed",
            MappingProxyType({symbol: bool(value) for symbol, value in self.ma_allowed.items()}),
        )
        object.__setattr__(
            self,
            "stratum_by_symbol",
            MappingProxyType(
                {symbol: (value[0], int(value[1]), int(value[2])) for symbol, value in self.stratum_by_symbol.items()}
            ),
        )


@dataclass(frozen=True)
class GateQuota:
    """门控对照必须精确匹配的旧仓和新仓数量。"""

    total: int
    retained: int
    new: int

    def __post_init__(self) -> None:
        if min(self.total, self.retained, self.new) < 0:
            raise ValueError("gate quota cannot be negative")
        if self.total != self.retained + self.new:
            raise ValueError("gate quota total must equal retained + new")


@dataclass(frozen=True)
class FrozenGateIdentities:
    """在任何成本重放前冻结的 FC/FMA 及其全部随机门身份。"""

    protocol_id: str
    rng_algorithm_version: str
    decision_dt: pd.Timestamp
    plan_sha256: str
    seeds: tuple[int, ...]
    fc_symbols: tuple[str, ...]
    fc_quota: GateQuota
    fma_symbols: tuple[str, ...]
    fma_quota: GateQuota
    fgr_symbols_by_seed: Mapping[int, tuple[str, ...]]
    fmgr_symbols_by_seed: Mapping[int, tuple[str, ...]]
    identity_sha256: str

    def __post_init__(self) -> None:
        if not self.protocol_id or self.rng_algorithm_version != RNG_ALGORITHM_VERSION:
            raise ValueError("frozen gate identities must bind the protocol and frozen RNG algorithm")
        seeds = tuple(map(int, self.seeds))
        if not seeds or len(seeds) != len(set(seeds)):
            raise ValueError("frozen gate identities require unique seeds")
        if set(map(int, self.fgr_symbols_by_seed)) != set(seeds) or set(map(int, self.fmgr_symbols_by_seed)) != set(
            seeds
        ):
            raise ValueError("FGR/FMGR seed maps must exactly match frozen seeds")
        object.__setattr__(self, "decision_dt", pd.Timestamp(self.decision_dt))
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "fc_symbols", tuple(map(str, self.fc_symbols)))
        object.__setattr__(self, "fma_symbols", tuple(map(str, self.fma_symbols)))
        object.__setattr__(
            self,
            "fgr_symbols_by_seed",
            MappingProxyType(
                {int(seed): tuple(map(str, symbols)) for seed, symbols in self.fgr_symbols_by_seed.items()}
            ),
        )
        object.__setattr__(
            self,
            "fmgr_symbols_by_seed",
            MappingProxyType(
                {int(seed): tuple(map(str, symbols)) for seed, symbols in self.fmgr_symbols_by_seed.items()}
            ),
        )
        for label, value in (("plan_sha256", self.plan_sha256), ("identity_sha256", self.identity_sha256)):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{label} must be a lowercase SHA256")
        if len(self.fc_symbols) != self.fc_quota.total or len(self.fma_symbols) != self.fma_quota.total:
            raise ValueError("gate symbol counts must match frozen quotas")
        if any(len(symbols) != self.fc_quota.total for symbols in self.fgr_symbols_by_seed.values()):
            raise ValueError("every FGR seed must match the FC quota")
        if any(len(symbols) != self.fma_quota.total for symbols in self.fmgr_symbols_by_seed.values()):
            raise ValueError("every FMGR seed must match the FMA quota")


@dataclass(frozen=True)
class SlotBudgetResult:
    """Reference-book slot resolution performed before gate quotas are frozen."""

    effective_plan: DecisionPlan
    blocked_outside_targets: tuple[str, ...]
    held_factor_targets: tuple[str, ...]
    admitted_new_targets: tuple[str, ...]


def canonical_json(value: Any) -> bytes:
    """稳定 JSON 编码，用于协议和决策指纹。"""

    def default(item: Any) -> Any:
        if isinstance(item, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(item).strftime("%Y-%m-%d")
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, (np.integer,)):
            return int(item)
        if isinstance(item, (np.floating,)):
            return None if np.isnan(item) else float(item)
        raise TypeError(f"cannot serialize {type(item)!r}")

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=default).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rng_for(*identity: Any) -> np.random.Generator:
    """Domain-separated deterministic RNG without additive seed/date collisions."""

    digest = hashlib.sha256(canonical_json(list(identity))).digest()
    entropy = np.frombuffer(digest, dtype="<u4")
    return np.random.default_rng(np.random.SeedSequence(entropy))


def load_protocol(path: Path = PROTOCOL_PATH) -> tuple[V2Config, dict[str, Any]]:
    """读取并严格校验冻结的 V2 协议；不接受宽松默认值。"""

    raw = path.read_bytes()
    frozen_raw = PROTOCOL_PATH.read_bytes()
    if raw != frozen_raw:
        raise ProtocolError(f"protocol must be a byte-for-byte copy of {PROTOCOL_PATH.name}")
    protocol = json.loads(raw)
    required = {
        "protocol_id",
        "protocol_status",
        "frequency",
        "execution",
        "selection",
        "factor_model",
        "timing_arms",
        "controls",
        "forward_validation",
        "decision_tree",
    }
    if missing := required - set(protocol):
        raise ProtocolError(f"protocol missing keys: {sorted(missing)}")
    exact = {
        "protocol_status": "LOCKED_PENDING_CLEAN_DATA",
        "frequency": "weekly_last_completed_trading_week",
        "execution": "decision_close_t_to_open_t_plus_1",
    }
    for key, expected in exact.items():
        if protocol.get(key) != expected:
            raise ProtocolError(f"unsupported {key}: {protocol.get(key)!r}; expected {expected!r}")

    selection = protocol["selection"]
    if selection.get("target_size") != 50 or selection.get("retention_max_rank") != 75:
        raise ProtocolError("V2 only supports the preregistered 50/75 selection buffer")
    ma = protocol["timing_arms"]["ma_placebo"]
    if ma.get("window") != 20 or ma.get("condition") != "adjusted_close_t_gt_sma20_t":
        raise ProtocolError("V2 only supports the preregistered SMA20 placebo")
    chan = protocol["timing_arms"]["chan"]
    if chan.get("state_columns_whitelist") != list(STATE_COLUMNS):
        raise ProtocolError("state whitelist must exactly match the physical projection")
    weights = protocol["factor_model"].get("weights")
    if weights != {"mom_120_20": 0.5, "lowvol_60": 0.5}:
        raise ProtocolError("V2 factor weights are frozen at 0.5/0.5")
    winsor = protocol["factor_model"].get("winsor_limits")
    if winsor != [0.01, 0.99]:
        raise ProtocolError("V2 winsor limits are frozen at 1%/99%")
    controls = protocol["controls"]
    decision = protocol["decision_tree"]
    statistics = protocol["statistics"]
    if controls.get("rng_algorithm_version") != RNG_ALGORITHM_VERSION:
        raise ProtocolError("V2 RNG algorithm version differs from the frozen contract")
    eligibility = protocol.get("eligibility", {})
    expected_eligibility = {
        "min_market_sessions_since_listing": 250,
        "min_valid_sessions_60": 57,
        "min_adv20_thousand_cny": 50000.0,
        "exclude_st": True,
        "exclude_boards": ["STAR", "BSE"],
        "canonical_board_values": ["MAIN", "CHINEXT", "STAR", "BSE"],
        "universe_timing": "list_date_lte_t_and_delist_date_gt_t",
        "missing_pit_field_policy": "ineligible",
    }
    if eligibility != expected_eligibility:
        raise ProtocolError("V2 eligibility policy differs from the frozen PIT universe contract")
    if protocol["factor_model"].get("winsor_quantile_method") != "linear":
        raise ProtocolError("V2 winsor quantile interpolation must be linear")
    if (
        protocol["forward_validation"].get("formal_start_rule")
        != "first_completed_week_with_eligible_count_gte_target_size_then_no_later_shortfall"
    ):
        raise ProtocolError("V2 formal start rule differs from the frozen contract")
    if protocol["forward_validation"].get("required_engineering_gates") != list(REQUIRED_ENGINEERING_GATES):
        raise ProtocolError("V2 engineering gate list differs from the frozen contract")
    if protocol["forward_validation"].get("dependency_sha256_keys") != list(FORWARD_DEPENDENCY_KEYS):
        raise ProtocolError("V2 dependency closure differs from the frozen contract")
    if (
        protocol["forward_validation"].get("local_ledger_scope")
        != "structural_integrity_only_never_self_sets_confirmatory_oos"
    ):
        raise ProtocolError("V2 local ledger scope differs from the frozen contract")
    if statistics.get("zero_sample_std_policy") != "invalidate_comparison":
        raise ProtocolError("V2 zero-standard-deviation policy must invalidate the comparison")
    if (
        statistics.get("bootstrap_rng_derivation")
        != "sha256_domain_separated_method_seed_comparison_identity_window_block_draws"
    ):
        raise ProtocolError("V2 bootstrap RNG derivation differs from the frozen contract")
    if statistics.get("confirmatory_comparisons") != list(CONFIRMATORY_COMPARISONS):
        raise ProtocolError("V2 confirmatory comparisons differ from the preregistered set")
    data_contract = protocol.get("data_contract", {})
    if data_contract.get("corporate_action_types") != list(CORPORATE_ACTION_TYPES):
        raise ProtocolError("V2 corporate action types differ from the frozen contract")
    if data_contract.get("universe_reconciliation_partitions") != list(UNIVERSE_RECONCILIATION_PARTITIONS):
        raise ProtocolError("V2 universe reconciliation partitions differ from the frozen contract")
    config = V2Config(
        protocol_id=str(protocol["protocol_id"]),
        protocol_status=str(protocol["protocol_status"]),
        selection=SelectionSpec(50, 75, 20),
        eligibility=EligibilitySpec(250, 57, 50000.0, ("STAR", "BSE")),
        allowed_chan_regimes=tuple(map(int, chan["allowed_regimes"])),
        random_seed=int(controls["random_seed"]),
        random_seed_count=int(controls["random_seed_count"]),
        winsor_low=float(winsor[0]),
        winsor_high=float(winsor[1]),
        neutralization_buckets=5,
        minimum_complete_weeks=int(protocol["forward_validation"]["minimum_complete_weeks"]),
        minimum_hac_t=float(decision["minimum_hac_t_one_sided"]),
        maximum_top10_share=float(decision["maximum_top10_positive_weeks_share"]),
        maximum_exposure_gap=float(decision["maximum_fc_fgr_average_exposure_gap"]),
        bootstrap_method=str(statistics["block_bootstrap_method"]),
        bootstrap_block_weeks=int(statistics["block_length_weeks"]),
        bootstrap_draws=int(statistics["bootstrap_draws"]),
        bootstrap_seed=int(statistics["bootstrap_seed"]),
        bootstrap_lower_quantile=float(statistics["bootstrap_lower_quantile"]),
    )
    return config, protocol


def derive_eligibility(features: pd.DataFrame, spec: EligibilitySpec) -> pd.DataFrame:
    """从可审计的 PIT 字段重算股票池，忽略任何调用方提供的 ``eligible``。"""

    required = {
        "symbol",
        "dt",
        "pit_universe_member",
        "market_sessions_since_listing",
        "valid_sessions_60",
        "is_st",
        "board",
        "adv20",
        "mom_120_20",
        "lowvol_60",
        "industry_code",
        "free_float_mcap",
        "adjusted_close",
        "sma20",
    }
    if missing := required - set(features):
        raise ValueError(f"eligibility fields missing: {sorted(missing)}")
    out = features.drop(columns=["eligible", "ineligibility_reasons"], errors="ignore").copy()
    for column in ("pit_universe_member", "is_st"):
        if out[column].isna().any() or not pd.api.types.is_bool_dtype(out[column].dtype):
            raise ValueError(f"{column} must be a complete boolean PIT field")
    if out.duplicated(["symbol", "dt"]).any():
        raise ValueError("duplicate symbol/dt before eligibility derivation")
    numeric_columns = [
        "market_sessions_since_listing",
        "valid_sessions_60",
        "adv20",
        "mom_120_20",
        "lowvol_60",
        "free_float_mcap",
        "adjusted_close",
        "sma20",
    ]
    numeric = out[numeric_columns].apply(pd.to_numeric, errors="coerce")
    complete_numeric = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    industry = out["industry_code"].astype("string")
    board = out["board"].astype("string")
    complete_text = industry.notna() & industry.str.len().gt(0) & board.notna() & board.str.len().gt(0)
    universe = out["pit_universe_member"].eq(True)
    seasoned = numeric["market_sessions_since_listing"].ge(spec.min_market_sessions_since_listing)
    sufficiently_traded = numeric["valid_sessions_60"].ge(spec.min_valid_sessions_60)
    valid_session_count = numeric["valid_sessions_60"].between(0, 60) & numeric["valid_sessions_60"].mod(1).eq(0)
    listing_session_count = numeric["market_sessions_since_listing"].ge(0) & numeric[
        "market_sessions_since_listing"
    ].mod(1).eq(0)
    liquid = numeric["adv20"].ge(spec.min_adv20_thousand_cny)
    non_st = out["is_st"].eq(False)
    board_allowed = ~board.isin(spec.excluded_boards)
    positive = (
        numeric["adv20"].gt(0)
        & numeric["free_float_mcap"].gt(0)
        & numeric["adjusted_close"].gt(0)
        & numeric["sma20"].gt(0)
    )
    checks = {
        "outside_pit_universe": universe,
        "listing_history_lt_min": seasoned,
        "valid_sessions_60_lt_min": sufficiently_traded,
        "adv20_lt_min": liquid,
        "st": non_st,
        "excluded_board": board_allowed,
        "missing_pit_or_factor": pd.Series(complete_numeric, index=out.index)
        & complete_text
        & positive
        & valid_session_count
        & listing_session_count,
    }
    checks = {reason: passed.fillna(False) for reason, passed in checks.items()}
    eligible = pd.Series(True, index=out.index)
    for passed in checks.values():
        eligible &= passed.fillna(False)
    out["eligible"] = eligible.astype(bool)
    out["ineligibility_reasons"] = [
        "|".join(reason for reason, passed in checks.items() if not bool(passed.iloc[index]))
        for index in range(len(out))
    ]
    return out


def compute_adjusted_features(
    prices: pd.DataFrame,
    calendar: Sequence[Any] | pd.DataFrame,
    *,
    asof: str | pd.Timestamp | None = None,
    ma_window: int = 20,
) -> pd.DataFrame:
    """On an exchange-session clock, carry prices through suspensions and set suspended amount to zero."""

    required = {"symbol", "dt", "close", "adj_factor", "amount"}
    if missing := required - set(prices):
        raise ValueError(f"price columns missing: {sorted(missing)}")
    frame = prices.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["dt"] = pd.to_datetime(frame["dt"])
    if frame.duplicated(["symbol", "dt"]).any():
        raise ValueError("duplicate symbol/dt in price track")
    for column in ("close", "adj_factor", "amount"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
    if not np.isfinite(frame[["close", "adj_factor", "amount"]].to_numpy()).all():
        raise ValueError("close, adj_factor and amount must be finite")
    if (frame[["close", "adj_factor"]] <= 0).any().any():
        raise ValueError("close and adj_factor must be positive")
    if (frame["amount"] < 0).any():
        raise ValueError("amount cannot be negative")
    if isinstance(calendar, pd.DataFrame):
        if "trade_date" not in calendar:
            raise ValueError("calendar must contain trade_date")
        sessions = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"]))
    else:
        sessions = pd.DatetimeIndex(pd.to_datetime(list(calendar)))
    if sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("calendar sessions must be non-empty, unique and increasing")
    if not frame["dt"].isin(sessions).all():
        raise ValueError("every observed price date must be an exchange session")
    cutoff = pd.Timestamp(asof) if asof is not None else pd.Timestamp(frame["dt"].max())
    if cutoff not in sessions:
        raise ValueError("asof must be an exchange session")
    if (frame["dt"] > cutoff).any():
        frame = frame[frame["dt"] <= cutoff].copy()
    if frame.empty:
        raise ValueError("no observed prices exist on or before asof")
    dense_parts: list[pd.DataFrame] = []
    for symbol, observed in frame.groupby("symbol", sort=True, observed=True):
        observed = observed.sort_values("dt", kind="mergesort").set_index("dt")
        symbol_sessions = sessions[(sessions >= observed.index.min()) & (sessions <= cutoff)]
        dense = observed.reindex(symbol_sessions)
        dense["symbol"] = str(symbol)
        dense["is_valid_trade"] = dense["close"].notna() & dense["amount"].fillna(0).gt(0)
        dense["close"] = dense["close"].ffill()
        dense["adj_factor"] = dense["adj_factor"].ffill()
        dense["amount"] = dense["amount"].fillna(0.0)
        if dense[["close", "adj_factor"]].isna().any().any():
            raise ValueError(f"cannot carry feature price before first observation for {symbol}")
        dense.index.name = "dt"
        dense_parts.append(dense.reset_index())
    frame = pd.concat(dense_parts, ignore_index=True).sort_values(["symbol", "dt"]).reset_index(drop=True)
    grouped = frame.groupby("symbol", sort=False, observed=True)
    frame["adjusted_close"] = frame["close"] * frame["adj_factor"]
    frame["log_adjusted_close"] = np.log(frame["adjusted_close"])
    log_grouped = frame.groupby("symbol", sort=False, observed=True)["log_adjusted_close"]
    frame["mom_120_20"] = log_grouped.shift(20) - log_grouped.shift(120)
    frame["log_return"] = log_grouped.diff()
    frame["lowvol_60"] = grouped["log_return"].transform(lambda values: -values.rolling(60, min_periods=60).std())
    frame["sma20"] = grouped["adjusted_close"].transform(
        lambda values: values.rolling(ma_window, min_periods=ma_window).mean()
    )
    frame["adv20"] = grouped["amount"].transform(lambda values: values.rolling(20, min_periods=20).mean())
    frame["valid_sessions_60"] = grouped["is_valid_trade"].transform(
        lambda values: values.astype(np.int16).rolling(60, min_periods=60).sum()
    )
    frame["ma_allowed"] = frame["adjusted_close"] > frame["sma20"]
    return frame.drop(columns=["log_adjusted_close", "log_return"])


def _percentile_bucket(values: pd.Series, symbols: pd.Series, buckets: int) -> pd.Series:
    """Return deterministic quantile buckets, using symbol only to break exact ties."""
    if buckets < 1:
        raise ValueError("buckets must be positive")
    count = min(int(buckets), len(values))
    if count <= 1:
        return pd.Series(np.zeros(len(values), dtype=np.int8), index=values.index)
    ordered = pd.DataFrame(
        {"value": pd.to_numeric(values, errors="raise").astype(float), "symbol": symbols.astype(str)},
        index=values.index,
    ).sort_values(["value", "symbol"], kind="mergesort")
    ordinal = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    return pd.qcut(ordinal, count, labels=False).astype("int8").reindex(values.index)


def _neutralized_rank(group: pd.DataFrame, factor: str, q_low: float, q_high: float) -> pd.Series:
    """对单日因子做冻结的行业 + log 自由流通市值 OLS 残差排名。"""

    values = pd.to_numeric(group[factor], errors="raise").astype(float)
    clipped = values.clip(
        values.quantile(q_low, interpolation="linear"),
        values.quantile(q_high, interpolation="linear"),
    )
    mcap = pd.to_numeric(group["free_float_mcap"], errors="raise").astype(float)
    if (mcap <= 0).any() or mcap.isna().any():
        raise ValueError("free_float_mcap must be complete and positive")
    industry = group["industry_code"].astype("string")
    if industry.isna().any() or (industry.str.len() == 0).any():
        raise ValueError("PIT industry_code must be complete")
    log_mcap = np.log(mcap.to_numpy())
    std = float(np.std(log_mcap, ddof=0))
    scaled_mcap = (log_mcap - float(np.mean(log_mcap))) / std if std > 0 else np.zeros(len(group))
    categories = sorted(industry.astype(str).unique())
    columns = [np.ones(len(group)), scaled_mcap]
    for category in categories[1:]:
        columns.append((industry.astype(str).to_numpy() == category).astype(float))
    design = np.column_stack(columns)
    beta, *_ = np.linalg.lstsq(design, clipped.to_numpy(), rcond=None)
    residual = clipped.to_numpy() - design @ beta
    return pd.Series(residual, index=group.index).rank(method="average", pct=True)


def rank_cross_section_v2(
    features: pd.DataFrame,
    q_low: float = 0.01,
    q_high: float = 0.99,
    buckets: int = 5,
) -> pd.DataFrame:
    """按日生成唯一的行业/规模残差化因子排名和对照联合分层。"""

    required = {
        "symbol",
        "dt",
        "eligible",
        "mom_120_20",
        "lowvol_60",
        "adv20",
        "industry_code",
        "free_float_mcap",
        "adjusted_close",
        "sma20",
    }
    if missing := required - set(features):
        raise ValueError(f"feature columns missing: {sorted(missing)}")
    if not (0 <= q_low < q_high <= 1):
        raise ValueError("invalid winsor limits")
    source = features.copy()
    source["symbol"] = source["symbol"].astype(str)
    source["dt"] = pd.to_datetime(source["dt"])
    if source["eligible"].isna().any() or not pd.api.types.is_bool_dtype(source["eligible"].dtype):
        raise ValueError("eligible must be a complete boolean derived field")
    source = source[source["eligible"].eq(True)].copy()
    if source.duplicated(["symbol", "dt"]).any():
        raise ValueError("duplicate symbol/dt in decision features")
    parts: list[pd.DataFrame] = []
    for decision_dt, group in source.groupby("dt", sort=True, observed=True):
        group = group.copy()
        formal_numeric = required - {"symbol", "dt", "industry_code", "eligible"}
        if group[list(formal_numeric)].isna().any().any():
            raise ValueError(f"missing formal feature at {pd.Timestamp(decision_dt).date()}")
        numeric_columns = list(formal_numeric)
        numeric = group[numeric_columns].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"non-finite formal feature at {pd.Timestamp(decision_dt).date()}")
        group["mom_120_20_rank"] = _neutralized_rank(group, "mom_120_20", q_low, q_high)
        group["lowvol_60_rank"] = _neutralized_rank(group, "lowvol_60", q_low, q_high)
        group["factor_score"] = 0.5 * group["mom_120_20_rank"] + 0.5 * group["lowvol_60_rank"]
        group["mcap_bucket"] = _percentile_bucket(group["free_float_mcap"], group["symbol"], buckets)
        group["adv_bucket"] = _percentile_bucket(group["adv20"], group["symbol"], buckets)
        group["ma_allowed"] = group["adjusted_close"] > group["sma20"]
        group = group.sort_values(["factor_score", "symbol"], ascending=[False, True], kind="mergesort")
        group["factor_rank"] = np.arange(1, len(group) + 1, dtype=np.int32)
        group["dt"] = pd.Timestamp(decision_dt)
        parts.append(group)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def attach_chan_gate(ranked: pd.DataFrame, states: pd.DataFrame, allowed: Sequence[int]) -> pd.DataFrame:
    """物理上仅接受三列状态投影，缺状态按 fail-closed 拒绝新仓。"""

    if tuple(states.columns) != STATE_COLUMNS:
        raise ValueError(f"state projection must be exactly {STATE_COLUMNS}, got {tuple(states.columns)}")
    projection = states.copy()
    projection["symbol"] = projection["symbol"].astype(str)
    projection["dt"] = pd.to_datetime(projection["dt"])
    numeric_regime = pd.to_numeric(projection["regime"], errors="raise")
    if not np.isfinite(numeric_regime.to_numpy(dtype=float)).all():
        raise ValueError("regime must be finite")
    if not np.equal(numeric_regime.to_numpy(dtype=float), np.floor(numeric_regime.to_numpy(dtype=float))).all():
        raise ValueError("regime must be integral")
    projection["regime"] = numeric_regime.astype("int8")
    if projection.duplicated(["symbol", "dt"]).any():
        raise ValueError("duplicate symbol/dt in state projection")
    if (~projection["regime"].between(0, 10)).any():
        raise ValueError("regime must be in 0..10")
    allowed_set = set(map(int, allowed))
    if not allowed_set or not allowed_set.issubset(set(range(11))):
        raise ValueError("allowed regimes must be a non-empty subset of 0..10")
    out = ranked.merge(projection, on=["symbol", "dt"], how="left", validate="one_to_one")
    out["chan_allowed"] = out["regime"].isin(allowed_set)
    return out


def select_with_rank_buffer(
    group: pd.DataFrame,
    previous_targets: Iterable[str],
    target_size: int,
    retention_max_rank: int,
) -> SelectionResult:
    """保留 rank<=退出线的旧目标，再按当期排名补足固定槽位。"""

    if target_size <= 0 or retention_max_rank < target_size:
        raise ValueError("selection requires 0 < target_size <= retention_max_rank")
    required = {"symbol", "factor_rank"}
    if missing := required - set(group):
        raise ValueError(f"ranking columns missing: {sorted(missing)}")
    ranked = group.loc[:, ["symbol", "factor_rank"]].copy()
    ranked["symbol"] = ranked["symbol"].astype(str)
    ranked["factor_rank"] = pd.to_numeric(ranked["factor_rank"], errors="raise").astype(int)
    if ranked["symbol"].duplicated().any() or ranked["factor_rank"].duplicated().any():
        raise ValueError("symbol and factor_rank must be unique within a decision")
    if len(ranked) < target_size:
        raise ValueError(f"eligible universe {len(ranked)} is smaller than target_size {target_size}")
    ranked = ranked.sort_values(["factor_rank", "symbol"], kind="mergesort")
    previous = set(map(str, previous_targets))
    retained = ranked[ranked["symbol"].isin(previous) & (ranked["factor_rank"] <= retention_max_rank)]
    retained_symbols = retained.head(target_size)["symbol"].tolist()
    selected = list(retained_symbols)
    selected_set = set(selected)
    for symbol in ranked["symbol"]:
        if symbol not in selected_set:
            selected.append(symbol)
            selected_set.add(symbol)
        if len(selected) == target_size:
            break
    reason = {symbol: ("buffer_retain" if symbol in set(retained_symbols) else "rank_entry") for symbol in selected}
    return SelectionResult(tuple(selected), reason)


def _stratum(row: pd.Series) -> tuple[str, int, int]:
    return str(row["industry_code"]), int(row["mcap_bucket"]), int(row["adv_bucket"])


def validate_weekly_schedule(schedule: pd.DataFrame, calendar: Sequence[Any] | pd.DataFrame) -> pd.DataFrame:
    """Verify that a supplied schedule is one contiguous subsequence of frozen calendar week boundaries."""

    if isinstance(calendar, pd.DataFrame):
        if "trade_date" not in calendar:
            raise ValueError("calendar must contain trade_date")
        sessions = pd.DatetimeIndex(pd.to_datetime(calendar["trade_date"]))
    else:
        sessions = pd.DatetimeIndex(pd.to_datetime(list(calendar)))
    if len(sessions) < 2 or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("calendar sessions must be unique, strictly increasing and contain at least two dates")
    normalized = schedule.copy()
    required = {"decision_dt", "exec_dt"}
    if missing := required - set(normalized):
        raise ValueError(f"schedule columns missing: {sorted(missing)}")
    normalized["decision_dt"] = pd.to_datetime(normalized["decision_dt"])
    normalized["exec_dt"] = pd.to_datetime(normalized["exec_dt"])
    if normalized.empty:
        raise ValueError("schedule cannot be empty")
    if normalized["decision_dt"].duplicated().any() or normalized["exec_dt"].duplicated().any():
        raise ValueError("schedule decision_dt and exec_dt must be unique")
    if not normalized["decision_dt"].is_monotonic_increasing:
        raise ValueError("schedule decisions must be strictly increasing")

    frozen_pairs = [
        (pd.Timestamp(current), pd.Timestamp(following))
        for current, following in zip(sessions[:-1], sessions[1:], strict=True)
        if current.isocalendar()[:2] != following.isocalendar()[:2]
    ]
    pair_to_index = {pair: index for index, pair in enumerate(frozen_pairs)}
    supplied = list(normalized[["decision_dt", "exec_dt"]].itertuples(index=False, name=None))
    try:
        positions = [
            pair_to_index[(pd.Timestamp(decision), pd.Timestamp(execution))] for decision, execution in supplied
        ]
    except KeyError as exc:
        raise ValueError("every decision must be the last session of its week and exec the next session") from exc
    if positions != list(range(positions[0], positions[0] + len(positions))):
        raise ValueError("schedule must not skip a complete trading week")
    return normalized


def build_buffered_target_plans(
    ranked: pd.DataFrame,
    schedule: pd.DataFrame,
    spec: SelectionSpec,
    all_adv: pd.DataFrame,
    calendar: Sequence[Any] | pd.DataFrame,
) -> dict[pd.Timestamp, DecisionPlan]:
    """逐期构建路径依赖但完全确定的 50/75 因子目标。"""

    schedule = validate_weekly_schedule(schedule, calendar)
    exec_map = schedule.set_index("decision_dt")["exec_dt"].to_dict()
    ranked_decisions = set(pd.to_datetime(ranked["dt"]))
    scheduled_decisions = set(exec_map)
    if ranked_decisions != scheduled_decisions:
        missing_ranked = sorted(
            pd.Timestamp(value).date().isoformat() for value in scheduled_decisions - ranked_decisions
        )
        unscheduled = sorted(pd.Timestamp(value).date().isoformat() for value in ranked_decisions - scheduled_decisions)
        raise ValueError(
            f"ranked/schedule decision dates differ; missing_ranked={missing_ranked}, unscheduled={unscheduled}"
        )

    adv = all_adv.loc[:, ["symbol", "dt", "adv20"]].copy()
    adv["symbol"] = adv["symbol"].astype(str)
    adv["dt"] = pd.to_datetime(adv["dt"])
    adv["adv20"] = pd.to_numeric(adv["adv20"], errors="raise").astype(float)
    if adv.duplicated(["symbol", "dt"]).any():
        raise ValueError("all_adv must be unique by symbol/dt")
    adv_by_dt = {pd.Timestamp(dt): frame.set_index("symbol")["adv20"] for dt, frame in adv.groupby("dt", sort=False)}

    plans: dict[pd.Timestamp, DecisionPlan] = {}
    previous_targets: tuple[str, ...] = ()
    for decision_dt, group in ranked.groupby("dt", sort=True, observed=True):
        decision = pd.Timestamp(decision_dt)
        if decision not in exec_map:
            continue
        selected = select_with_rank_buffer(group, previous_targets, spec.target_size, spec.retention_max_rank)
        indexed = group.set_index("symbol", drop=False)
        current_adv = adv_by_dt.get(decision)
        if current_adv is None:
            raise ValueError(f"missing all-stock ADV snapshot for {decision.date()}")
        ranked_adv = pd.to_numeric(indexed["adv20"], errors="raise").astype(float)
        missing_pool_adv = sorted(symbol for symbol in ranked_adv.index if symbol not in current_adv)
        if missing_pool_adv:
            raise ValueError(f"all_adv omits ranked symbols at {decision.date()}: {missing_pool_adv[:5]}")
        aligned_adv = current_adv.reindex(ranked_adv.index).astype(float)
        if not np.allclose(ranked_adv.to_numpy(), aligned_adv.to_numpy(), rtol=1e-12, atol=1e-9):
            raise ValueError(f"ranked/all_adv values differ at {decision.date()}")
        required_adv = set(selected.symbols) | set(previous_targets)
        missing_adv = sorted(
            symbol
            for symbol in required_adv
            if symbol not in current_adv or not np.isfinite(current_adv[symbol]) or float(current_adv[symbol]) <= 0
        )
        if missing_adv:
            raise ValueError(f"missing current decision ADV for exits/targets at {decision.date()}: {missing_adv[:5]}")
        plans[decision] = DecisionPlan(
            decision_dt=decision,
            exec_dt=pd.Timestamp(exec_map[decision]),
            symbols=selected.symbols,
            rank_by_symbol={symbol: int(indexed.loc[symbol, "factor_rank"]) for symbol in selected.symbols},
            selection_reason=dict(selected.reason_by_symbol),
            adv_cny={symbol: float(current_adv[symbol]) * 1000.0 for symbol in required_adv},
            chan_allowed={symbol: bool(indexed.loc[symbol, "chan_allowed"]) for symbol in selected.symbols},
            ma_allowed={symbol: bool(indexed.loc[symbol, "ma_allowed"]) for symbol in selected.symbols},
            stratum_by_symbol={symbol: _stratum(indexed.loc[symbol]) for symbol in selected.symbols},
        )
        previous_targets = selected.symbols
    return plans


def derive_gate_decision(
    plan: DecisionPlan,
    held_symbols: Iterable[str],
    gate: str,
) -> tuple[tuple[str, ...], GateQuota]:
    """冻结真实门控身份；只有实际已持有股票免重新入场门。"""

    if gate == "chan":
        allowed = plan.chan_allowed
    elif gate == "ma":
        allowed = plan.ma_allowed
    else:
        raise ValueError(f"unsupported gate: {gate}")
    held = set(map(str, held_symbols))
    retained = tuple(symbol for symbol in plan.symbols if symbol in held)
    accepted_new = tuple(symbol for symbol in plan.symbols if symbol not in held and bool(allowed.get(symbol, False)))
    accepted_set = set(retained) | set(accepted_new)
    accepted = tuple(symbol for symbol in plan.symbols if symbol in accepted_set)
    return accepted, GateQuota(total=len(accepted), retained=len(retained), new=len(accepted_new))


def apply_reference_slot_budget(
    plan: DecisionPlan,
    *,
    actual_held_symbols: Iterable[str],
    blocked_exit_symbols: Iterable[str],
    maximum_positions: int = 50,
) -> SlotBudgetResult:
    """Let blocked exits occupy slots and deterministically trim new candidates before any gate."""

    if maximum_positions <= 0:
        raise ValueError("maximum_positions must be positive")
    held = set(map(str, actual_held_symbols))
    blocked = set(map(str, blocked_exit_symbols))
    if not blocked.issubset(held):
        raise ValueError("blocked exits must be a subset of actual reference holdings")
    if len(held) > maximum_positions:
        raise ValueError("reference book already exceeds the maximum position count")
    factor_set = set(plan.symbols)
    blocked_outside = tuple(sorted(blocked - factor_set))
    held_targets = tuple(symbol for symbol in plan.symbols if symbol in held)
    available_new_slots = maximum_positions - len(blocked_outside) - len(held_targets)
    if available_new_slots < 0:
        raise ValueError("blocked exits and held factor targets exceed the slot budget")
    admitted_new = tuple(symbol for symbol in plan.symbols if symbol not in held)[:available_new_slots]
    admitted = set(held_targets) | set(admitted_new)
    effective_symbols = tuple(symbol for symbol in plan.symbols if symbol in admitted)
    effective = DecisionPlan(
        decision_dt=plan.decision_dt,
        exec_dt=plan.exec_dt,
        symbols=effective_symbols,
        rank_by_symbol={symbol: plan.rank_by_symbol[symbol] for symbol in effective_symbols},
        selection_reason={symbol: plan.selection_reason[symbol] for symbol in effective_symbols},
        adv_cny=dict(plan.adv_cny),
        chan_allowed={symbol: plan.chan_allowed[symbol] for symbol in effective_symbols},
        ma_allowed={symbol: plan.ma_allowed[symbol] for symbol in effective_symbols},
        stratum_by_symbol={symbol: plan.stratum_by_symbol[symbol] for symbol in effective_symbols},
    )
    if len(effective_symbols) + len(blocked_outside) > maximum_positions:
        raise AssertionError("slot resolution exceeded maximum positions")
    return SlotBudgetResult(effective, blocked_outside, held_targets, admitted_new)


def apply_matched_random_gate(
    base_symbols: Sequence[str],
    held_symbols: Iterable[str],
    quota: GateQuota,
    seed: int,
    *,
    protocol_id: str,
    arm: str,
    seed_index: int,
    decision_dt: pd.Timestamp | str,
) -> tuple[str, ...]:
    """精确匹配真实门的 retained/new 配额；非法配额不静默截断。"""

    base = tuple(map(str, base_symbols))
    if len(base) != len(set(base)):
        raise ValueError("base_symbols must be unique")
    held = set(map(str, held_symbols))
    retained = tuple(symbol for symbol in base if symbol in held)
    new_candidates = tuple(symbol for symbol in base if symbol not in held)
    if len(retained) != quota.retained:
        raise ControlMatchError(f"retained quota {quota.retained} does not match actual {len(retained)}")
    if quota.new > len(new_candidates):
        raise ControlMatchError(f"new quota {quota.new} exceeds candidate count {len(new_candidates)}")
    if not protocol_id or arm not in {"FGR", "FMGR"} or seed_index < 0:
        raise ValueError("matched gate requires a protocol id, FGR/FMGR arm, and non-negative seed index")
    decision = pd.Timestamp(decision_dt)
    rng = _rng_for(
        RNG_ALGORITHM_VERSION,
        protocol_id,
        arm,
        int(seed_index),
        decision.strftime("%Y-%m-%d"),
        int(seed),
        base,
        sorted(held),
        quota.total,
        quota.retained,
        quota.new,
    )
    chosen_new = set(rng.choice(new_candidates, size=quota.new, replace=False)) if quota.new else set()
    accepted_set = set(retained) | chosen_new
    accepted = tuple(symbol for symbol in base if symbol in accepted_set)
    if len(accepted) != quota.total:
        raise AssertionError("matched random gate failed exact quota")
    return accepted


def freeze_gate_identities(
    plan: DecisionPlan,
    *,
    held_fc: Iterable[str],
    held_fma: Iterable[str],
    held_fgr_by_seed: Mapping[int, Iterable[str]],
    held_fmgr_by_seed: Mapping[int, Iterable[str]],
    config: V2Config,
) -> FrozenGateIdentities:
    """Freeze all timing identities from actual reference books before gross/1x/2x replay."""

    frozen_seeds = tuple(range(config.random_seed, config.random_seed + config.random_seed_count))
    held_fgr = {int(seed): tuple(symbols) for seed, symbols in held_fgr_by_seed.items()}
    held_fmgr = {int(seed): tuple(symbols) for seed, symbols in held_fmgr_by_seed.items()}
    if set(held_fgr) != set(frozen_seeds) or set(held_fmgr) != set(frozen_seeds):
        raise ControlMatchError("held control books must exactly cover every frozen seed")
    fc_symbols, fc_quota = derive_gate_decision(plan, held_fc, "chan")
    fma_symbols, fma_quota = derive_gate_decision(plan, held_fma, "ma")
    fgr = {
        seed: apply_matched_random_gate(
            plan.symbols,
            held_fgr[seed],
            fc_quota,
            seed,
            protocol_id=config.protocol_id,
            arm="FGR",
            seed_index=index,
            decision_dt=plan.decision_dt,
        )
        for index, seed in enumerate(frozen_seeds)
    }
    fmgr = {
        seed: apply_matched_random_gate(
            plan.symbols,
            held_fmgr[seed],
            fma_quota,
            seed,
            protocol_id=config.protocol_id,
            arm="FMGR",
            seed_index=index,
            decision_dt=plan.decision_dt,
        )
        for index, seed in enumerate(frozen_seeds)
    }
    plan_hash = hashlib.sha256(canonical_json(plan_to_dict(plan))).hexdigest()
    identity_payload = {
        "protocol_id": config.protocol_id,
        "rng_algorithm_version": RNG_ALGORITHM_VERSION,
        "decision_dt": plan.decision_dt,
        "plan_sha256": plan_hash,
        "seeds": list(frozen_seeds),
        "fc_symbols": list(fc_symbols),
        "fc_quota": {"total": fc_quota.total, "retained": fc_quota.retained, "new": fc_quota.new},
        "fma_symbols": list(fma_symbols),
        "fma_quota": {"total": fma_quota.total, "retained": fma_quota.retained, "new": fma_quota.new},
        "fgr_symbols_by_seed": {str(seed): list(fgr[seed]) for seed in frozen_seeds},
        "fmgr_symbols_by_seed": {str(seed): list(fmgr[seed]) for seed in frozen_seeds},
    }
    identity_hash = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
    return FrozenGateIdentities(
        protocol_id=config.protocol_id,
        rng_algorithm_version=RNG_ALGORITHM_VERSION,
        decision_dt=plan.decision_dt,
        plan_sha256=plan_hash,
        seeds=frozen_seeds,
        fc_symbols=fc_symbols,
        fc_quota=fc_quota,
        fma_symbols=fma_symbols,
        fma_quota=fma_quota,
        fgr_symbols_by_seed=fgr,
        fmgr_symbols_by_seed=fmgr,
        identity_sha256=identity_hash,
    )


def build_matched_random_plans(
    ranked: pd.DataFrame,
    factor_plans: Mapping[pd.Timestamp, DecisionPlan],
    all_adv: pd.DataFrame,
    seed: int,
    *,
    protocol_id: str,
    seed_index: int,
) -> dict[pd.Timestamp, DecisionPlan]:
    """按行业×市值×ADV 联合层和每层留存数精确构建 R_match。"""

    adv = all_adv.loc[:, ["symbol", "dt", "adv20"]].copy()
    adv["symbol"] = adv["symbol"].astype(str)
    adv["dt"] = pd.to_datetime(adv["dt"])
    adv["adv20"] = pd.to_numeric(adv["adv20"], errors="raise").astype(float)
    if adv.duplicated(["symbol", "dt"]).any():
        raise ValueError("all_adv must be unique by symbol/dt")
    adv_by_dt = {pd.Timestamp(dt): frame.set_index("symbol")["adv20"] for dt, frame in adv.groupby("dt", sort=False)}
    groups = {pd.Timestamp(dt): frame.copy() for dt, frame in ranked.groupby("dt", sort=False, observed=True)}
    previous_factor: set[str] = set()
    previous_random: set[str] = set()
    out: dict[pd.Timestamp, DecisionPlan] = {}

    for decision, factor_plan in sorted(factor_plans.items()):
        group = groups.get(pd.Timestamp(decision))
        if group is None:
            raise ControlMatchError(f"missing ranked pool for {pd.Timestamp(decision).date()}")
        group = group.copy()
        group["stratum"] = group.apply(_stratum, axis=1)
        indexed = group.set_index("symbol", drop=False)
        target_counts = Counter(factor_plan.stratum_by_symbol.values())
        target_retained: Counter[tuple[str, int, int]] = Counter(
            factor_plan.stratum_by_symbol[symbol] for symbol in factor_plan.symbols if symbol in previous_factor
        )
        if not protocol_id or seed_index < 0:
            raise ValueError("R_match requires a protocol id and non-negative seed index")
        rng = _rng_for(
            RNG_ALGORITHM_VERSION,
            protocol_id,
            "R_MATCH",
            int(seed_index),
            pd.Timestamp(decision).strftime("%Y-%m-%d"),
            int(seed),
        )
        selected: list[str] = []
        retained_selected: set[str] = set()

        for stratum in sorted(target_counts):
            in_stratum = group["stratum"].map(lambda value, target=stratum: value == target)
            pool = sorted(group.loc[in_stratum, "symbol"].astype(str))
            incumbents = [symbol for symbol in pool if symbol in previous_random]
            rng.shuffle(incumbents)
            need_retained = int(target_retained[stratum])
            if len(incumbents) < need_retained:
                raise ControlMatchError(
                    f"{pd.Timestamp(decision).date()} stratum {stratum} has {len(incumbents)} random incumbents, "
                    f"needs {need_retained}"
                )
            kept = incumbents[:need_retained]
            selected.extend(kept)
            retained_selected.update(kept)
            need_new = int(target_counts[stratum]) - need_retained
            candidates = [symbol for symbol in pool if symbol not in previous_random and symbol not in selected]
            if len(candidates) < need_new:
                raise ControlMatchError(
                    f"{pd.Timestamp(decision).date()} stratum {stratum} has {len(candidates)} new controls, needs {need_new}"
                )
            if need_new:
                chosen = rng.choice(candidates, size=need_new, replace=False)
                selected.extend(map(str, chosen))

        if len(selected) != len(factor_plan.symbols) or len(selected) != len(set(selected)):
            raise ControlMatchError(f"invalid matched selection at {pd.Timestamp(decision).date()}")
        selected = sorted(selected, key=lambda symbol: (_stratum(indexed.loc[symbol]), symbol))
        current_adv = adv_by_dt.get(pd.Timestamp(decision))
        if current_adv is None:
            raise ControlMatchError(f"missing ADV snapshot at {pd.Timestamp(decision).date()}")
        if "adv20" not in indexed:
            raise ControlMatchError("ranked pool must retain adv20 for source-consistency checks")
        ranked_adv = pd.to_numeric(indexed["adv20"], errors="raise").astype(float)
        if any(symbol not in current_adv for symbol in ranked_adv.index):
            raise ControlMatchError(f"all_adv omits ranked controls at {pd.Timestamp(decision).date()}")
        if not np.allclose(
            ranked_adv.to_numpy(),
            current_adv.reindex(ranked_adv.index).astype(float).to_numpy(),
            rtol=1e-12,
            atol=1e-9,
        ):
            raise ControlMatchError(f"ranked/all_adv values differ at {pd.Timestamp(decision).date()}")
        required_adv = set(selected) | previous_random
        missing_adv = sorted(
            symbol
            for symbol in required_adv
            if symbol not in current_adv or not np.isfinite(current_adv[symbol]) or float(current_adv[symbol]) <= 0
        )
        if missing_adv:
            raise ControlMatchError(f"missing exit/target ADV for R_match: {missing_adv[:5]}")
        out[pd.Timestamp(decision)] = DecisionPlan(
            decision_dt=pd.Timestamp(decision),
            exec_dt=factor_plan.exec_dt,
            symbols=tuple(selected),
            rank_by_symbol={symbol: int(indexed.loc[symbol, "factor_rank"]) for symbol in selected},
            selection_reason={
                symbol: ("matched_retain" if symbol in retained_selected else "matched_new") for symbol in selected
            },
            adv_cny={symbol: float(current_adv[symbol]) * 1000.0 for symbol in required_adv},
            chan_allowed=dict.fromkeys(selected, True),
            ma_allowed=dict.fromkeys(selected, True),
            stratum_by_symbol={symbol: _stratum(indexed.loc[symbol]) for symbol in selected},
        )
        if Counter(out[pd.Timestamp(decision)].stratum_by_symbol.values()) != target_counts:
            raise AssertionError("R_match stratum counts drifted")
        actual_retained = Counter(
            out[pd.Timestamp(decision)].stratum_by_symbol[symbol] for symbol in selected if symbol in previous_random
        )
        if actual_retained != target_retained:
            raise AssertionError("R_match retention counts drifted")
        previous_factor = set(factor_plan.symbols)
        previous_random = set(selected)
    return out


def factor_diagnostics_v2(ranked: pd.DataFrame, labels: pd.DataFrame, topn: int) -> pd.DataFrame:
    """名单先冻结再合并未来标签；Top-N 参数不允许硬编码回 Top20。"""

    if topn <= 0:
        raise ValueError("topn must be positive")
    required_labels = {"symbol", "dt", "forward_return"}
    if missing := required_labels - set(labels):
        raise ValueError(f"label columns missing: {sorted(missing)}")
    future = labels.copy()
    future["symbol"] = future["symbol"].astype(str)
    future["dt"] = pd.to_datetime(future["dt"])
    if future.duplicated(["symbol", "dt"]).any():
        raise ValueError("labels must be unique by symbol/dt")
    rows: list[dict[str, Any]] = []
    for decision, group in ranked.groupby("dt", sort=True, observed=True):
        frozen = group.nsmallest(topn, "factor_rank").loc[:, ["symbol"]].copy()
        labeled = group.merge(future, on=["symbol", "dt"], how="left", validate="one_to_one")
        frozen_labeled = frozen.merge(
            future[future["dt"].eq(pd.Timestamp(decision))], on="symbol", how="left", validate="one_to_one"
        )
        valid = labeled.dropna(subset=["forward_return"])
        universe_coverage = float(labeled["forward_return"].notna().mean())
        topn_coverage = float(frozen_labeled["forward_return"].notna().mean())
        finite_labels = np.isfinite(valid["forward_return"].to_numpy(dtype=float)).all()
        period_valid = universe_coverage == 1.0 and topn_coverage == 1.0 and bool(finite_labels)
        row = {
            "decision_dt": pd.Timestamp(decision),
            "n_universe": int(len(group)),
            "n_labeled": int(len(valid)),
            "label_coverage": universe_coverage,
            "period_valid": period_valid,
            "invalid_reason": None if period_valid else "missing_or_non_finite_forward_label_no_survivor_drop",
            "score_rank_ic": labeled["factor_score"].corr(labeled["forward_return"], method="spearman")
            if period_valid
            else np.nan,
            "momentum_rank_ic": labeled["mom_120_20_rank"].corr(labeled["forward_return"], method="spearman")
            if period_valid
            else np.nan,
            "lowvol_rank_ic": labeled["lowvol_60_rank"].corr(labeled["forward_return"], method="spearman")
            if period_valid
            else np.nan,
            "topn": int(topn),
            "topn_equal_weight_return": frozen_labeled["forward_return"].mean() if period_valid else np.nan,
            "topn_label_coverage": topn_coverage,
        }
        if period_valid and len(labeled) >= 5:
            labeled = labeled.copy()
            labeled["score_quintile"] = pd.qcut(labeled["factor_score"].rank(method="first"), 5, labels=False) + 1
            for quintile, value in labeled.groupby("score_quintile", observed=True)["forward_return"].mean().items():
                row[f"score_q{int(quintile)}_return"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_seed_weekly_v2(
    weekly_by_seed: Mapping[int, pd.Series],
    config: V2Config,
) -> pd.Series:
    """Freeze the preregistered operation order: equal-mean 20 seeds per week, then statistics."""

    expected_seeds = tuple(range(config.random_seed, config.random_seed + config.random_seed_count))
    normalized = {int(seed): values for seed, values in weekly_by_seed.items()}
    if set(normalized) != set(expected_seeds):
        raise ValueError("weekly seed returns must exactly cover all 20 preregistered seeds")
    first_index: pd.Index | None = None
    columns: list[pd.Series] = []
    for seed in expected_seeds:
        series = pd.Series(normalized[seed], dtype=float)
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(f"seed {seed} weekly index must be unique and increasing")
        if first_index is None:
            first_index = series.index
        elif not series.index.equals(first_index):
            raise ValueError("all seed paths must have the exact same weekly index")
        if len(series) != config.minimum_complete_weeks or not np.isfinite(series.to_numpy()).all():
            raise ValueError("each seed path must contain exactly 52 finite weekly returns")
        columns.append(series.rename(seed))
    return pd.concat(columns, axis=1).mean(axis=1).rename("equal_seed_mean")


def _newey_west_t_weekly(values: np.ndarray, max_lag: int = 4) -> float:
    count = len(values)
    centered = values - float(values.mean())
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, min(max_lag, count - 1) + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        long_run_variance += 2.0 * (1.0 - lag / (max_lag + 1.0)) * covariance
    if long_run_variance <= 0:
        return 0.0
    return float(values.mean() / math.sqrt(long_run_variance / count))


def _circular_block_bootstrap_lower(
    values: np.ndarray,
    *,
    block_weeks: int,
    draws: int,
    lower_quantile: float,
    seed: int,
    identity: str,
) -> float:
    if block_weeks <= 0 or draws <= 0 or not (0 < lower_quantile < 0.5):
        raise ValueError("invalid frozen bootstrap configuration")
    count = len(values)
    blocks = math.ceil(count / block_weeks)
    rng = _rng_for("circular_block_bootstrap_v2", int(seed), str(identity), count, block_weeks, draws)
    starts = rng.integers(0, count, size=(draws, blocks))
    offsets = np.arange(block_weeks)
    indices = (starts[..., None] + offsets) % count
    samples = values[indices.reshape(draws, -1)[:, :count]]
    annualized_means = samples.mean(axis=1) * 52.0
    return float(np.quantile(annualized_means, lower_quantile, method="linear"))


def active_statistics_v2(
    weekly_active: pd.Series | Sequence[float], config: V2Config, identity: str
) -> dict[str, float]:
    """Compute the only preregistered first-52-week active-return statistics."""

    if identity not in CONFIRMATORY_COMPARISONS:
        raise ValueError(f"unregistered confirmatory comparison: {identity}")
    series = pd.Series(weekly_active, dtype=float)
    if len(series) != config.minimum_complete_weeks:
        raise ValueError("confirmatory statistics require exactly the first 52 complete weeks")
    values = series.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("weekly active returns must be finite; invalid periods cannot be dropped")
    positives = np.sort(values[values > 0])[::-1]
    positive_sum = float(positives.sum())
    top10_share = float(positives[:10].sum() / positive_sum) if positive_sum > 0 else 0.0
    weekly_std = float(values.std(ddof=1))
    if weekly_std <= 0:
        raise ValueError("zero weekly sample standard deviation invalidates the confirmatory comparison")
    return {
        "annualized_active_arithmetic": float(values.mean() * 52.0),
        "weekly_active_median": float(np.median(values)),
        "top10_positive_weeks_share": top10_share,
        "hac_t_weekly_mean_lag4": _newey_west_t_weekly(values, max_lag=4),
        "block_bootstrap_lower": _circular_block_bootstrap_lower(
            values,
            block_weeks=config.bootstrap_block_weeks,
            draws=config.bootstrap_draws,
            lower_quantile=config.bootstrap_lower_quantile,
            seed=config.bootstrap_seed,
            identity=identity,
        ),
        "information_ratio": float(math.sqrt(52.0) * values.mean() / weekly_std),
    }


def ir_mdd_improvement_v2(
    fc_weekly_returns: pd.Series | Sequence[float],
    f_weekly_returns: pd.Series | Sequence[float],
    config: V2Config,
) -> dict[str, float | bool]:
    """Compute the frozen FC-vs-F IR-or-drawdown improvement rule on the first 52 weeks."""

    fc = pd.Series(fc_weekly_returns, dtype=float)
    factor = pd.Series(f_weekly_returns, dtype=float)
    if len(fc) != config.minimum_complete_weeks or len(factor) != config.minimum_complete_weeks:
        raise ValueError("IR/MDD comparison requires exactly 52 weeks")
    if (
        not fc.index.equals(factor.index)
        or not np.isfinite(fc.to_numpy()).all()
        or not np.isfinite(factor.to_numpy()).all()
    ):
        raise ValueError("FC and F weekly paths must have identical indices and finite returns")
    if (fc <= -1).any() or (factor <= -1).any():
        raise ValueError("weekly return cannot be less than or equal to -100%")

    def summarize(values: pd.Series) -> tuple[float, float]:
        std = float(values.std(ddof=1))
        if std <= 0:
            raise ValueError("zero weekly sample standard deviation invalidates IR/MDD comparison")
        information_ratio = float(math.sqrt(52.0) * values.mean() / std)
        nav = np.concatenate(([1.0], (1.0 + values).cumprod().to_numpy(dtype=float)))
        maximum_drawdown = float(np.min(nav / np.maximum.accumulate(nav) - 1.0))
        return information_ratio, maximum_drawdown

    fc_ir, fc_mdd = summarize(fc)
    f_ir, f_mdd = summarize(factor)
    return {
        "fc_information_ratio": fc_ir,
        "f_information_ratio": f_ir,
        "fc_maximum_drawdown": fc_mdd,
        "f_maximum_drawdown": f_mdd,
        "FC_improves_ir_or_drawdown": fc_ir > f_ir or fc_mdd > f_mdd,
    }


def _metric(metrics: Mapping[str, Any], comparison: str, field: str) -> float:
    try:
        value = metrics[comparison][field]
    except KeyError as exc:
        raise ValueError(f"missing metric {comparison}.{field}") from exc
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite metric {comparison}.{field}")
    return value


def _evaluate_v2_logic(
    readiness: Mapping[str, Any],
    engineering: Mapping[str, Any],
    controls: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: V2Config,
) -> dict[str, Any]:
    """Pure decision-tree helper; callers may not treat mapping inputs as verified evidence."""

    if readiness.get("forward_start_allowed") is not True or engineering.get("all_passed") is not True:
        return {
            "status": "INVALID_DATA_OR_ENGINEERING",
            "alpha_validated": False,
            "failed_data_gates": list(readiness.get("failed_gates", [])),
            "failed_engineering_checks": list(engineering.get("failed_checks", [])),
        }
    if readiness.get("confirmatory_oos") is not True:
        return {
            "status": "FORWARD_COLLECTION_REQUIRED",
            "alpha_validated": False,
            "complete_weeks": int(readiness.get("complete_weeks", 0)),
            "required_weeks": config.minimum_complete_weeks,
        }
    exposure_gap = float(controls.get("fc_fgr_average_exposure_gap", math.inf))
    expected_seeds = list(range(config.random_seed, config.random_seed + config.random_seed_count))
    control_checks = {
        "all_exact": controls.get("all_exact") is True,
        "r_match_seeds_exact": controls.get("r_match_seed_ids") == expected_seeds,
        "fgr_seeds_exact": controls.get("fgr_seed_ids") == expected_seeds,
        "fmgr_seeds_exact": controls.get("fmgr_seed_ids") == expected_seeds,
        "unselected_equal_seed_mean": controls.get("seed_aggregation")
        == "per_week_equal_seed_mean_then_compute_preregistered_statistics",
        "identity_frozen_before_cost_replay": controls.get("identity_frozen_before_cost_replay") is True,
        "same_identity_all_cost_scenarios": controls.get("same_identity_all_cost_scenarios") is True,
        "first_frozen_window_only": controls.get("first_frozen_window_only") is True,
        "exact_window_length": int(controls.get("evaluation_window_weeks", -1)) == config.minimum_complete_weeks,
        "all_weekly_identities_present": int(controls.get("gate_identity_hash_count", -1))
        == config.minimum_complete_weeks,
        "no_invalid_period": int(controls.get("invalid_period_count", -1)) == 0,
        "exposure_gap_bounded": math.isfinite(exposure_gap) and 0.0 <= exposure_gap <= config.maximum_exposure_gap,
    }
    if not all(control_checks.values()):
        return {
            "status": "INVALID_CONTROL",
            "alpha_validated": False,
            "fc_fgr_average_exposure_gap": exposure_gap,
            "maximum_allowed": config.maximum_exposure_gap,
            "control_checks": control_checks,
        }

    factor_name = "F_2x_minus_R_match_2x"
    factor_checks = {
        "annualized_active_positive": _metric(metrics, factor_name, "annualized_active_arithmetic") > 0,
        "weekly_median_positive": _metric(metrics, factor_name, "weekly_active_median") > 0,
        "top10_share_bounded": _metric(metrics, factor_name, "top10_positive_weeks_share")
        <= config.maximum_top10_share,
        "hac_t_pass": _metric(metrics, factor_name, "hac_t_weekly_mean_lag4") >= config.minimum_hac_t,
        "bootstrap_lower_positive": _metric(metrics, factor_name, "block_bootstrap_lower") > 0,
    }
    if not all(factor_checks.values()):
        return {
            "status": "REJECT_FACTOR",
            "alpha_validated": False,
            "factor_checks": factor_checks,
            "timing_evaluated": False,
        }

    identity = "FC_gross_minus_FGR_gross"
    cost_identity = "FC_2x_minus_FGR_2x"
    placebo = "FC_gross_minus_FMA_gross"
    cost_placebo = "FC_2x_minus_FMA_2x"
    economic_checks = {
        "identity_gross_positive": _metric(metrics, identity, "annualized_active_arithmetic") > 0,
        "identity_2x_positive": _metric(metrics, cost_identity, "annualized_active_arithmetic") > 0,
        "ma_placebo_positive": _metric(metrics, placebo, "annualized_active_arithmetic") > 0,
        "ma_placebo_2x_positive": _metric(metrics, cost_placebo, "annualized_active_arithmetic") > 0,
        "improves_ir_or_drawdown": metrics.get("FC_improves_ir_or_drawdown") is True,
    }
    if not all(economic_checks.values()):
        return {
            "status": "REJECT_TIMING",
            "alpha_validated": False,
            "factor_checks": factor_checks,
            "timing_economic_checks": economic_checks,
        }
    statistical_checks = {
        "identity_hac_t_pass": _metric(metrics, identity, "hac_t_weekly_mean_lag4") >= config.minimum_hac_t,
        "identity_bootstrap_lower_positive": _metric(metrics, identity, "block_bootstrap_lower") > 0,
        "placebo_hac_t_pass": _metric(metrics, placebo, "hac_t_weekly_mean_lag4") >= config.minimum_hac_t,
        "placebo_bootstrap_lower_positive": _metric(metrics, placebo, "block_bootstrap_lower") > 0,
    }
    if not all(statistical_checks.values()):
        return {
            "status": "INCONCLUSIVE",
            "alpha_validated": False,
            "factor_checks": factor_checks,
            "timing_economic_checks": economic_checks,
            "timing_statistical_checks": statistical_checks,
        }
    return {
        "status": "FORWARD_EVIDENCE_PASSED_SHADOW_ONLY",
        "alpha_validated": False,
        "live_trading_allowed": False,
        "factor_checks": factor_checks,
        "timing_economic_checks": economic_checks,
        "timing_statistical_checks": statistical_checks,
    }


def evaluate_v2(
    readiness: Mapping[str, Any],
    engineering: Mapping[str, Any],
    controls: Mapping[str, Any],
    metrics: Mapping[str, Any],
    config: V2Config,
) -> dict[str, Any]:
    """Fail closed until a semantic replay verifier can construct formal evidence.

    The mappings remain accepted only for API migration and test diagnostics;
    they are not an authority boundary.  Once the replay verifier exists, this
    entry point must read its content-addressed evidence manifest and recompute
    the first-52-week statistics rather than trusting these values.
    """

    del readiness, engineering, controls, metrics, config
    return {
        "status": "INVALID_DATA_OR_ENGINEERING",
        "alpha_validated": False,
        "live_trading_allowed": False,
        "failed_engineering_checks": ["semantic_artifact_replay_verifier_unavailable"],
    }


def plan_to_dict(plan: DecisionPlan) -> dict[str, Any]:
    return {
        "decision_dt": plan.decision_dt,
        "exec_dt": plan.exec_dt,
        "symbols": list(plan.symbols),
        "rank_by_symbol": dict(plan.rank_by_symbol),
        "selection_reason": dict(plan.selection_reason),
        "adv_cny": dict(plan.adv_cny),
        "chan_allowed": dict(plan.chan_allowed),
        "ma_allowed": dict(plan.ma_allowed),
        "stratum_by_symbol": {symbol: list(value) for symbol, value in plan.stratum_by_symbol.items()},
    }


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        file.write(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("protocol-check", help="严格校验并打印冻结协议指纹")
    check.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)

    plan = subparsers.add_parser("plan", help="从干净的决策特征生成缓冲后名单和 R_match")
    plan.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    plan.add_argument("--features", type=Path, required=True)
    plan.add_argument("--states", type=Path, required=True)
    plan.add_argument("--schedule", type=Path, required=True)
    plan.add_argument("--calendar", type=Path, required=True)
    plan.add_argument("--all-adv", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config, protocol = load_protocol(args.protocol)
    if args.command == "protocol-check":
        print(
            json.dumps(
                {
                    "protocol_id": config.protocol_id,
                    "status": config.protocol_status,
                    "protocol_sha256": sha256_file(args.protocol),
                    "engine_sha256": sha256_file(Path(__file__).resolve()),
                    "forward_start_allowed": False,
                    "reason": "clean immutable data and the semantic replay/verification stack are not attached",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    features = derive_eligibility(pd.read_parquet(args.features), config.eligibility)
    states = pd.read_parquet(args.states, columns=list(STATE_COLUMNS))
    schedule = pd.read_parquet(args.schedule)
    calendar = pd.read_parquet(args.calendar)
    all_adv = pd.read_parquet(args.all_adv)
    ranked = rank_cross_section_v2(features, config.winsor_low, config.winsor_high, config.neutralization_buckets)
    ranked = attach_chan_gate(ranked, states, config.allowed_chan_regimes)
    factor_plans = build_buffered_target_plans(ranked, schedule, config.selection, all_adv, calendar)
    matched_plan_sets = {
        config.random_seed + offset: build_matched_random_plans(
            ranked,
            factor_plans,
            all_adv,
            config.random_seed + offset,
            protocol_id=config.protocol_id,
            seed_index=offset,
        )
        for offset in range(config.random_seed_count)
    }
    identity = {
        "protocol": protocol,
        "engine_sha256": sha256_file(Path(__file__).resolve()),
        "features_sha256": sha256_file(args.features),
        "states_sha256": sha256_file(args.states),
        "schedule_sha256": sha256_file(args.schedule),
        "calendar_sha256": sha256_file(args.calendar),
        "all_adv_sha256": sha256_file(args.all_adv),
    }
    run_id = hashlib.sha256(canonical_json(identity)).hexdigest()[:16]
    output = args.output_dir / f"{RESEARCH_LABEL}_{run_id}"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing V2 artifact directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    ranked.to_parquet(output / "ranked.parquet", index=False)
    _write_exclusive(output / "identity.json", json.dumps(identity, ensure_ascii=False, indent=2).encode() + b"\n")
    _write_exclusive(
        output / "factor_plans.json",
        json.dumps(
            [plan_to_dict(plan) for _, plan in sorted(factor_plans.items())], ensure_ascii=False, indent=2, default=str
        ).encode()
        + b"\n",
    )
    _write_exclusive(
        output / "r_match_plans.json",
        json.dumps(
            {
                str(seed): [plan_to_dict(plan) for _, plan in sorted(plans.items())]
                for seed, plans in matched_plan_sets.items()
            },
            ensure_ascii=False,
            indent=2,
            default=str,
        ).encode()
        + b"\n",
    )
    print(output)


if __name__ == "__main__":
    main()
