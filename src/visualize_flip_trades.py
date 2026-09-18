#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Tool for Flip Strategy Trades (Real Slippage Aware).
- Interactively select Stock/Date/Session from backtest results.
- Loads raw mid-price data from .npz files for the background curve.
- **Update**: Uses 'entry_price' from the trade CSV for marker Y-coordinates.
  This visualizes the Real Slippage (Buy at Ask > Mid, Sell at Bid < Mid).
"""

try:
    from .configuration import load_config
except ImportError:
    from configuration import load_config

import argparse
import json
import logging
import os
import sys
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
except ImportError:
    print("Error: 'plotly' module not found. Please install it using: pip install plotly")
    sys.exit(1)

from typing import Dict

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def abspath(root: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root, p)

def safe_format(tpl: str, **kwargs) -> str:
    try: return tpl.format(**kwargs)
    except: return tpl

def load_jsonl_map(path: str) -> Dict[str, str]:
    m = {}
    if not os.path.exists(path):
        return m
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            obj = json.loads(s)
            key = f"{obj.get('stock_code')}_{obj.get('date')}_{obj.get('session')}"
            lp = obj.get("final_label_path") or obj.get("raw_label_path")
            if lp: m[key] = lp
    return m

# ---------------------------------------------------------
# Visualization Logic
# ---------------------------------------------------------

def plot_session(cfg: dict, trades_df: pd.DataFrame, stock: str, date: str, session: int):
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]
    # 1. Load Raw Price Data (Mid) from NPZ
    # We still plot Mid as the reference baseline.
    logging.info(f"Loading raw data for {stock} {date} Sess {session}...")
    labels_index_path = abspath(project_root, safe_format(cfg["stage2"]["labels_index_path"], horizon_id=horizon_id))
    label_map = load_jsonl_map(labels_index_path)

    map_key = f"{stock}_{date}_{session}"
    npz_rel_path = label_map.get(map_key)

    if not npz_rel_path:
        logging.error(f"Could not find .npz file path for key: {map_key}")
        return

    npz_path = abspath(project_root, npz_rel_path)
    try:
        z = np.load(npz_path)
        mid_raw = z["mid"].astype(np.float32)
        t_sec = z["t_sec"] if "t_sec" in z else np.arange(len(mid_raw))
    except Exception as e:
        logging.error(f"Failed to load {npz_path}: {e}")
        return

    # 2. Filter trades
    session_trades = trades_df[
        (trades_df["stock"].astype(str) == str(stock)) &
        (trades_df["date"].astype(str) == str(date)) &
        (trades_df["session"].astype(int) == int(session))
    ].copy()

    if session_trades.empty:
        logging.warning("No trades found for this session.")
        return

    logging.info(f"Found {len(session_trades)} trade segments. Generating plot...")

    # 3. Build Plotly Figure
    fig = go.Figure()

    # -- Base Layer: Mid Price Curve --
    fig.add_trace(go.Scattergl(
        x=t_sec,
        y=mid_raw,
        mode='lines',
        name='Mid Price',
        line=dict(color='rgba(100, 100, 100, 0.5)', width=1),
        hoverinfo='y'
    ))

    # -- Overlay Layer: Actual Execution Prices --

    long_entries = session_trades[session_trades["direction"] == 1]
    short_entries = session_trades[session_trades["direction"] == -1]

    # Plot LONG Entries
    if not long_entries.empty:
        # X: Time (from index)
        # Y: Actual Entry Price (from CSV, which is Ask)
        lx = long_entries["start_time_sec"].values
        ly = long_entries["entry_price"].values # <--- KEY CHANGE: Use recorded price

        fig.add_trace(go.Scattergl(
            x=lx, y=ly,
            mode='markers',
            name='Long Entry (Ask)',
            marker=dict(symbol='triangle-up', color='green', size=10, line=dict(width=1, color='darkgreen')),
            hovertemplate="Time: %{x}<br>Buy Price: %{y:.3f}<extra></extra>"
        ))

    # Plot SHORT Entries
    if not short_entries.empty:
        # X: Time
        # Y: Actual Entry Price (from CSV, which is Bid)
        sx = short_entries["start_time_sec"].values
        sy = short_entries["entry_price"].values # <--- KEY CHANGE: Use recorded price

        fig.add_trace(go.Scattergl(
            x=sx, y=sy,
            mode='markers',
            name='Short Entry (Bid)',
            marker=dict(symbol='triangle-down', color='red', size=10, line=dict(width=1, color='darkred')),
            hovertemplate="Time: %{x}<br>Sell Price: %{y:.3f}<extra></extra>"
        ))

    # Layout
    fig.update_layout(
        title=f"Flip Strategy Execution (Real Slippage): {stock} - {date} - Sess {session}",
        xaxis_title="Time (Seconds from midnight)",
        yaxis_title="Price",
        hovermode="closest", # Changed from unified to separate points to see spread better
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        dragmode='zoom'
    )

    logging.info("Opening browser...")
    fig.show()

# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config_path)
    project_root = cfg["project"]["project_root"]
    horizon_id = cfg["horizons"]["active_horizon_id"]

    bt_cfg = cfg.get("backtest", {})
    run_name = bt_cfg.get("run_name") or cfg.get("predict", {}).get("run_name") or "unknown_run"
    out_map = bt_cfg.get("output", {})

    # Path logic matching backtest_engine_flip.py
    base_csv = out_map.get("trade_detail_csv", "backtest/trades.csv")
    csv_fmt = safe_format(base_csv, horizon_id=horizon_id, run_name=run_name)
    trade_csv_path = abspath(project_root, csv_fmt)

    if not os.path.exists(trade_csv_path):
        legacy_path = abspath(project_root, csv_fmt.replace(".csv", "_flip.csv"))
        if os.path.exists(legacy_path):
            trade_csv_path = legacy_path
        else:
            logging.error(f"Trade file not found: {trade_csv_path}")
            sys.exit(1)

    logging.info(f"Loading trades: {trade_csv_path}")
    df_trades = pd.read_csv(trade_csv_path)
    if "start_time_sec" not in df_trades.columns:
        logging.error("Trade file predates row-aligned V3 output; rerun the backtest.")
        sys.exit(1)
    df_trades['date'] = df_trades['date'].astype(str)

    choices = df_trades.groupby(['stock', 'date', 'session']).size().reset_index(name='count')

    while True:
        print("\n" + "="*60)
        print(f"{'ID':<4} | {'Stock':<10} | {'Date':<10} | {'Session':<6} | {'Count':<6}")
        print("-" * 60)
        for i, row in choices.head(20).iterrows():
            print(f"{i:<4} | {row['stock']:<10} | {row['date']:<10} | {row['session']:<6} | {row['count']:<6}")
        if len(choices) > 20: print("...")
        print("="*60)

        sel = input("\nEnter ID (q to quit): ").strip().lower()
        if sel == 'q': break
        try:
            idx = int(sel)
            if 0 <= idx < len(choices):
                r = choices.iloc[idx]
                plot_session(cfg, df_trades, r['stock'], r['date'], r['session'])
            else: print("Invalid ID.")
        except: print("Invalid input.")

if __name__ == "__main__":
    main()
