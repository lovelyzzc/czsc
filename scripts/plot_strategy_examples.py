"""笔趋势动量加速策略 — 买卖点示例 K 线图

用 plotly 绘制缠论结构 + 大号买卖标注，输出自包含 HTML。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from czsc import CZSC, CzscStrategyBase, Event, Position, format_standard_kline

OUTPUT_DIR = Path(__file__).resolve().parent / "_output" / "strategy_examples"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path.home() / ".ts_data_cache" / "a_stock_daily_qfq"
FREQ = "日线"

_ZDT = f"{FREQ}_D1_涨跌停V230331_涨停_任意_任意_0"
_BI_DOWN = f"{FREQ}_D1_表里关系V230101_向下_任意_任意_0"
_BI_UPTREND = f"{FREQ}_D4N1笔趋势_高低点辅助判断V230913_上升趋势_任意_任意_0"
_MACD_STRONG = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_强势_任意_任意_0"
_PRICE_ACC = f"{FREQ}_D1W10_加速V221110_上涨_任意_任意_0"
_MACD_WEAK = f"{FREQ}_D1K#MACD12#26#9强弱_BS辅助V221108_弱势_任意_任意_0"


class BiTrendAccelStrategy(CzscStrategyBase):
    @property
    def positions(self) -> list[Position]:
        return [Position(
            name="笔趋势动量加速_多头", symbol=self.symbol,
            opens=[Event.load({
                "name": "笔上升+MACD强势+加速_开多", "operate": "开多",
                "signals_all": [_BI_UPTREND, _MACD_STRONG, _PRICE_ACC],
                "signals_not": [_ZDT],
            })],
            exits=[
                Event.load({"name": "笔向下_平多", "operate": "平多",
                            "signals_all": [_BI_DOWN]}),
                Event.load({"name": "MACD转弱_平多", "operate": "平多",
                            "signals_all": [_MACD_WEAK]}),
            ],
            interval=3600 * 24 * 5, timeout=120, stop_loss=500, t0=False,
        )]

    @property
    def signals_config(self):
        base = list(super().signals_config)
        names = {c["name"] for c in base}
        extras = [
            {"name": "cxt_bi_trend_V230913", "freq": FREQ, "di": 4, "n": 1},
            {"name": "tas_macd_power_V221108", "freq": FREQ, "di": 1,
             "fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
            {"name": "bar_accelerate_V221110", "freq": FREQ, "di": 1, "window": 10},
        ]
        for cfg in extras:
            if cfg["name"] not in names:
                base.append(cfg)
        return base


def _load_bars(parquet_path: Path):
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None
    if len(df) < 500:
        return None
    df = df.rename(columns={"ts_code": "symbol", "trade_date": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    try:
        return format_standard_kline(df, freq=FREQ)
    except Exception:
        return None


def _run_strategy(bars):
    symbol = bars[0].symbol
    sdt = bars[len(bars) // 4].dt.strftime("%Y-%m-%d")
    strategy = BiTrendAccelStrategy(symbol=symbol)
    res = strategy.backtest(bars, sdt=sdt)
    return res.pairs_df(), sdt


def _generate_plotly_chart(bars, pairs: pd.DataFrame, symbol: str, tail_n: int = 200):
    """用 plotly 生成带大号买卖标注的缠论结构 K 线图"""
    c = CZSC(bars)

    # 取最近 tail_n 根 K 线
    raw_bars = c.bars_raw
    if tail_n and len(raw_bars) > tail_n:
        cutoff_dt = raw_bars[-tail_n].dt
        raw_bars = [b for b in raw_bars if b.dt >= cutoff_dt]
    else:
        cutoff_dt = raw_bars[0].dt

    dt_list = [b.dt for b in raw_bars]
    opens = [b.open for b in raw_bars]
    highs = [b.high for b in raw_bars]
    lows = [b.low for b in raw_bars]
    closes = [b.close for b in raw_bars]
    vols = [b.vol for b in raw_bars]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.15, 0.25],
        vertical_spacing=0.02,
    )

    # === K 线 ===
    colors = ["#e74c3c" if c_ >= o else "#27ae60" for o, c_ in zip(opens, closes)]
    fig.add_trace(go.Candlestick(
        x=dt_list, open=opens, high=highs, low=lows, close=closes,
        increasing_line_color="#e74c3c", decreasing_line_color="#27ae60",
        increasing_fillcolor="#e74c3c", decreasing_fillcolor="#27ae60",
        name="K线", showlegend=False,
    ), row=1, col=1)

    # === 笔（BI）===
    bi_list = c.bi_list
    bi_dt = [bi.fx_a.dt for bi in bi_list if bi.fx_a.dt >= cutoff_dt]
    bi_val = [bi.fx_a.fx for bi in bi_list if bi.fx_a.dt >= cutoff_dt]
    if bi_list:
        last_bi = bi_list[-1]
        if last_bi.fx_b.dt >= cutoff_dt:
            bi_dt.append(last_bi.fx_b.dt)
            bi_val.append(last_bi.fx_b.fx)

    if bi_dt:
        fig.add_trace(go.Scatter(
            x=bi_dt, y=bi_val,
            mode="lines", line=dict(color="#1a237e", width=2),
            name="笔", showlegend=True,
        ), row=1, col=1)

    # === 分型 (FX) ===
    fx_list = c.fx_list
    for fx in fx_list:
        if fx.dt < cutoff_dt:
            continue
        color = "#e74c3c" if str(fx.mark) == "Mark.G" else "#27ae60"
        fig.add_trace(go.Scatter(
            x=[fx.dt], y=[fx.fx],
            mode="markers", marker=dict(color=color, size=6, symbol="diamond"),
            showlegend=False,
        ), row=1, col=1)

    # === 成交量 ===
    fig.add_trace(go.Bar(
        x=dt_list, y=vols,
        marker_color=colors, name="成交量", showlegend=False,
    ), row=2, col=1)

    # === MACD ===
    from czsc._native.ta import sma as _sma
    close_arr = [b.close for b in c.bars_raw]
    import numpy as np
    c_arr = np.array(close_arr, dtype=float)
    ema12 = pd.Series(c_arr).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(c_arr).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd = 2 * (dif - dea)

    all_dt = [b.dt for b in c.bars_raw]
    # 截取到 tail 范围
    if tail_n and len(all_dt) > tail_n:
        start_idx = len(all_dt) - tail_n
    else:
        start_idx = 0

    macd_dt = all_dt[start_idx:]
    macd_dif = dif[start_idx:]
    macd_dea = dea[start_idx:]
    macd_bar = macd[start_idx:]

    macd_colors = ["#e74c3c" if v >= 0 else "#27ae60" for v in macd_bar]
    fig.add_trace(go.Bar(
        x=macd_dt, y=macd_bar, marker_color=macd_colors,
        name="MACD", showlegend=False,
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=macd_dt, y=macd_dif, mode="lines",
        line=dict(color="#e67e22", width=1), name="DIF",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=macd_dt, y=macd_dea, mode="lines",
        line=dict(color="#2980b9", width=1), name="DEA",
    ), row=3, col=1)

    # === 买卖点标注（大号箭头 + 文字）===
    visible_pairs = pairs[pairs["开仓时间"].apply(
        lambda x: pd.Timestamp(x) >= cutoff_dt
    )].reset_index(drop=True)

    buy_dt, buy_price, buy_text = [], [], []
    sell_dt, sell_price, sell_text = [], [], []

    for idx, row in visible_pairs.iterrows():
        open_dt = pd.Timestamp(row["开仓时间"])
        close_dt = pd.Timestamp(row["平仓时间"])
        open_price = row.get("开仓价格", 0)
        close_price = row.get("平仓价格", 0)
        profit_bp = row.get("盈亏比例", 0)
        pct = profit_bp / 100

        buy_dt.append(open_dt)
        buy_price.append(open_price)
        buy_text.append(f"B{idx+1}")

        sell_dt.append(close_dt)
        sell_price.append(close_price)
        profit_str = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
        sell_text.append(f"S{idx+1} {profit_str}")

        # 买卖连线（持仓区间）
        line_color = "#e74c3c" if pct > 0 else "#95a5a6"
        fig.add_trace(go.Scatter(
            x=[open_dt, close_dt], y=[open_price, close_price],
            mode="lines", line=dict(color=line_color, width=1.5, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=1)

    # 买入标记
    if buy_dt:
        fig.add_trace(go.Scatter(
            x=buy_dt, y=buy_price, mode="markers+text",
            marker=dict(symbol="triangle-up", size=18, color="#e74c3c",
                        line=dict(color="#fff", width=2)),
            text=buy_text, textposition="bottom center",
            textfont=dict(size=14, color="#e74c3c", family="IBM Plex Mono"),
            name="买入 (B)", showlegend=True,
        ), row=1, col=1)

    # 卖出标记
    if sell_dt:
        sell_colors = ["#27ae60" if (visible_pairs.iloc[i].get("盈亏比例", 0) > 0) else "#95a5a6"
                       for i in range(len(sell_dt))]
        fig.add_trace(go.Scatter(
            x=sell_dt, y=sell_price, mode="markers+text",
            marker=dict(symbol="triangle-down", size=18, color=sell_colors,
                        line=dict(color="#fff", width=2)),
            text=sell_text, textposition="top center",
            textfont=dict(size=12, color="#2c3e50", family="IBM Plex Mono"),
            name="卖出 (S)", showlegend=True,
        ), row=1, col=1)

    # === 布局 ===
    fig.update_layout(
        title=dict(
            text=f"<b>{symbol}</b>  笔趋势动量加速 — 买卖点示例",
            font=dict(size=18, family="IBM Plex Sans"),
        ),
        height=800, template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=40, t=80, b=40),
        font=dict(family="IBM Plex Sans"),
    )
    fig.update_xaxes(type="category", row=1, col=1)
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_xaxes(type="category", row=3, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)

    # 构造交易统计 HTML 头部
    total = len(visible_pairs)
    wins = int((visible_pairs["盈亏比例"] > 0).sum()) if total > 0 and "盈亏比例" in visible_pairs.columns else 0
    avg_ret = visible_pairs["盈亏比例"].mean() / 100 if total > 0 and "盈亏比例" in visible_pairs.columns else 0
    win_pct = int(wins / total * 100) if total > 0 else 0

    trade_info = f"""
    <div style="font-family:'IBM Plex Sans','PingFang SC',sans-serif;max-width:1000px;margin:0 auto;padding:16px 24px;">
      <div style="display:flex;gap:24px;align-items:center;flex-wrap:wrap;margin-bottom:8px;">
        <div style="font-size:13px;color:#8a8580;">
          <b>信号组合:</b> 笔上升趋势 + MACD 强势 + 价格加速 &nbsp;|&nbsp;
          <b>出场:</b> 笔向下 / MACD 转弱 &nbsp;|&nbsp;
          <b>可见交易:</b> {total} 笔 &nbsp;|&nbsp;
          <b>胜率:</b> {wins}/{total} ({win_pct}%) &nbsp;|&nbsp;
          <b>均收益:</b> {avg_ret:+.1f}%
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;">"""

    for idx, row in visible_pairs.iterrows():
        profit_bp = row.get("盈亏比例", 0)
        pct = profit_bp / 100
        bg = "#e8f5e9" if pct > 0 else "#ffebee"
        clr = "#2e7d32" if pct > 0 else "#c62828"
        pct_str = f"+{pct:.1f}%" if pct > 0 else f"{pct:.1f}%"
        trade_info += f"""
        <span style="display:inline-flex;align-items:center;gap:4px;padding:4px 10px;
          border-radius:6px;background:{bg};font-family:'IBM Plex Mono',monospace;font-size:13px;">
          <span style="color:#e74c3c;font-weight:700;">B{idx+1}</span>
          <span style="color:#8a8580;">→</span>
          <span style="color:{clr};font-weight:700;">S{idx+1} {pct_str}</span>
        </span>"""

    trade_info += """
      </div>
    </div>"""

    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head>
<meta charset="utf-8"/>
<title>{symbol} 笔趋势动量加速 — 买卖点</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>body{{margin:0;background:#faf8f5;font-family:'IBM Plex Sans',sans-serif;}}</style>
</head><body>
{trade_info}
{chart_html}
</body></html>"""

    return full_html


def main():
    CANDIDATE_CODES = [
        "000001.SZ", "600519.SH", "000858.SZ", "601318.SH", "600036.SH",
        "000333.SZ", "002475.SZ", "300750.SZ", "601888.SH", "000568.SZ",
        "002594.SZ", "600276.SH", "000725.SZ", "601012.SH", "002714.SZ",
        "600809.SH", "002352.SZ", "000661.SZ", "603259.SH", "601899.SH",
    ]

    print(f"扫描候选股票 ...")
    scored = []
    for code in CANDIDATE_CODES:
        p = DATA_DIR / f"{code}.parquet"
        if not p.exists():
            continue
        bars = _load_bars(p)
        if bars is None:
            continue
        pairs, sdt = _run_strategy(bars)
        if pairs.empty:
            continue
        n = len(pairs)
        wr = (pairs["盈亏比例"] > 0).mean() if "盈亏比例" in pairs.columns else 0
        scored.append({"code": code, "bars": bars, "pairs": pairs, "n": n, "wr": wr})
        print(f"  {code}: {n} 笔 | 胜率 {wr:.0%}")

    print(f"\n扫描全库找多笔交易标的 ...")
    existing = {s["code"] for s in scored}
    for pf in sorted(DATA_DIR.glob("*.parquet")):
        code = pf.stem
        if code in existing:
            continue
        bars = _load_bars(pf)
        if bars is None:
            continue
        pairs, sdt = _run_strategy(bars)
        if pairs.empty or len(pairs) < 3:
            continue
        n = len(pairs)
        wr = (pairs["盈亏比例"] > 0).mean() if "盈亏比例" in pairs.columns else 0
        scored.append({"code": code, "bars": bars, "pairs": pairs, "n": n, "wr": wr})
        if n >= 3:
            print(f"  {code}: {n} 笔 | 胜率 {wr:.0%}")

    scored.sort(key=lambda x: (-x["n"], -x["wr"]))
    selected = scored[:6]

    print(f"\n生成 {len(selected)} 只股票的 plotly K 线图 ...")
    for info in selected:
        code = info["code"]
        safe = code.replace(".", "_")
        html = _generate_plotly_chart(info["bars"], info["pairs"], code, tail_n=200)
        out = OUTPUT_DIR / f"{safe}_买卖点.html"
        out.write_text(html, encoding="utf-8")
        print(f"  {code}: {out.name} ({out.stat().st_size/1024:.1f} KB) "
              f"| {info['n']} 笔 | 胜率 {info['wr']:.0%}")

    print(f"\n[完成] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
