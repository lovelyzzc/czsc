"""Python 集成测试：验证 Rust iter_regime_states 行为正确性。"""

from czsc import CZSC, Freq, format_standard_kline
from czsc._native.trend_regime import (
    FeatureSnapshot,
    Regime,
    StateSnapshot,
    iter_regime_states,
    py_priority_score,
    py_surge_onset,
    py_surge_score,
)
from czsc.mock import generate_symbol_kines


def _make_bars(symbol="000001", freq_str="日线", start="20200101", end="20240101"):
    df = generate_symbol_kines(symbol, freq_str, start, end)
    return format_standard_kline(df, freq=Freq.D), df


class TestRegime:
    def test_from_int_roundtrip(self):
        for v in range(11):
            assert int(Regime.from_int(v)) == v

    def test_equality(self):
        assert Regime.MainUptrend == Regime.MainUptrend
        assert Regime.Downtrend != Regime.MainUptrend


class TestIterRegimeStates:
    def test_basic_output_shape(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, False, None, 9.8, 120, 6)
        assert len(states) > 0
        for s in states:
            assert isinstance(s, StateSnapshot)
            assert 0 <= s.regime <= 10
            assert 0 <= s.prev_regime <= 10
            assert s.close > 0

    def test_with_features(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, True, None, 9.8, 120, 6)
        has_feats = any(s.feats is not None for s in states)
        assert has_feats, "with_features=True should produce at least some FeatureSnapshot"
        for s in states:
            if s.feats is not None:
                assert isinstance(s.feats, FeatureSnapshot)
                assert s.feats.n_pivots >= 0

    def test_tail_mode(self):
        bars, _ = _make_bars()
        full = iter_regime_states(bars, Freq.D, False, None, 9.8, 120, 6)
        tail20 = iter_regime_states(bars, Freq.D, False, 20, 9.8, 120, 6)
        assert len(tail20) <= 20
        assert len(tail20) < len(full)

    def test_short_data_returns_empty(self):
        bars, _ = _make_bars(start="20240101", end="20240201")
        states = iter_regime_states(bars, Freq.D, False, None, 9.8, 120, 6)
        assert states == []

    def test_regime_distribution_reasonable(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, False, None, 9.8, 120, 6)
        regimes = {s.regime for s in states}
        assert 1 in regimes, "Should have at least Downtrend"
        assert any(r > 1 for r in regimes), "Should have states beyond Downtrend"

    def test_causal_consistency(self):
        """流式 CZSC 与全量构造的 bi_list 一致性抽样验证。"""
        bars, _ = _make_bars()
        n = len(bars)
        warmup = 120
        if n <= warmup:
            return

        czsc = CZSC(bars[:warmup])
        samples = list(range(warmup, n, max(1, (n - warmup) // 20)))[:20]

        for idx in range(warmup, n):
            czsc.update(bars[idx])
            if idx in samples:
                ref = CZSC(bars[: idx + 1]).bi_list
                cur = czsc.bi_list

                def _key(bl):
                    return [(b.sdt, b.edt, round(b.high, 4), round(b.low, 4)) for b in bl]

                assert _key(cur) == _key(ref), f"Causal leak at idx={idx}"


class TestSurgeOnset:
    def test_no_feats_returns_false(self):
        assert not py_surge_onset(1, 7, None, [], "confirm")
        assert not py_surge_onset(1, 5, None, [], "anticipate")

    def test_confirm_requires_feats(self):
        assert not py_surge_onset(1, 7, None, [4, 5], "confirm")


class TestSurgeScore:
    def test_none_returns_zero(self):
        assert py_surge_score(None) == 0.0


class TestPriorityScore:
    def test_basic(self):
        p = py_priority_score(50.0, 10.0, 0, 7, 10)
        assert p > 0

    def test_nan_sl(self):
        p = py_priority_score(50.0, float("nan"), 0, 7, 10)
        assert p >= 0


class TestFeatureSnapshot:
    def test_getters(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, True, None, 9.8, 120, 6)
        feats = [s.feats for s in states if s.feats is not None]
        assert len(feats) > 0
        f = feats[0]
        assert isinstance(f.n_pivots, int)
        assert isinstance(f.above_zg, bool)
        assert isinstance(f.ret20, float)

    def test_to_dict(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, True, None, 9.8, 120, 6)
        feats = [s.feats for s in states if s.feats is not None]
        if feats:
            d = feats[0].to_dict()
            assert isinstance(d, dict)
            assert "vol_ratio" in d
            assert "ret20" in d
            assert "above_zg" in d

    def test_get_method(self):
        bars, _ = _make_bars()
        states = iter_regime_states(bars, Freq.D, True, None, 9.8, 120, 6)
        feats = [s.feats for s in states if s.feats is not None]
        if feats:
            f = feats[0]
            assert f.get("n_pivots") is not None
            assert f.get("nonexistent") is None
