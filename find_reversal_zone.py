"""
find_reversal_zone.py

After a breakdown (price crosses below Delta Support and keeps falling), this
finds the zone where price actually found buyers and reversed - using a mini
volume profile (POC + Value Area) computed just around the swing low, rather
than the whole day. This is often a more tradeable support level than the
original (now-broken) Delta Support, since it shows real buying absorption.

Usage:
    python find_reversal_zone.py SYMBOL [DATE]

    SYMBOL - e.g. INDIGO
    DATE   - optional, YYYY-MM-DD. Defaults to most recent trading day in data.

SETUP: $env:UPSTOX_ACCESS_TOKEN = "your_token_here"
"""

import os
import sys
from datetime import datetime, timedelta

import requests
import pandas as pd

ACCESS_TOKEN = os.environ.get("UPSTOX_ACCESS_TOKEN", "")
UNIT = "minutes"
INTERVAL_VALUE = "5"
LOOKBACK_CALENDAR_DAYS = 15
VALUE_AREA_PCT = 0.70  # standard 70% value area


def resolve_equity_instrument_key(symbol):
    headers = {"Content-Type": "application/json", "Accept": "application/json",
               "Authorization": f"Bearer {ACCESS_TOKEN}"}
    params = {"query": symbol, "exchanges": "NSE", "segments": "EQ", "page_number": 1, "records": 10}
    resp = requests.get("https://api.upstox.com/v2/instruments/search", headers=headers, params=params, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    for inst in payload.get("data", []):
        if inst.get("trading_symbol", "").upper() == symbol.upper() and inst.get("instrument_type") == "EQ":
            return inst["instrument_key"]
    return None


def fetch_candles(instrument_key, to_date=None, from_date=None):
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    if from_date is None:
        from_date = (datetime.now() - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{UNIT}/{INTERVAL_VALUE}/{to_date}/{from_date}"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {ACCESS_TOKEN}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    candles = payload.get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["date"] = df["timestamp"].dt.date
    return df


def find_swing_low_window(day_df):
    """Find the swing low (lowest close) and return a window of bars around
    it covering the decline into the low and the initial recovery out of it -
    i.e. the basing/reversal period, not the whole day."""
    day_df = day_df.reset_index(drop=True)
    low_idx = day_df["low"].idxmin()
    low_price = day_df.loc[low_idx, "low"]
    low_time = day_df.loc[low_idx, "timestamp"]

    recovery_threshold = low_price * 1.005
    min_window_bars = 12  # ensure at least ~1 hour of context even on a fast rejection

    start_idx = max(0, low_idx - 12)
    end_idx = low_idx
    for i in range(low_idx, len(day_df)):
        end_idx = i
        if day_df.loc[i, "close"] >= recovery_threshold and (end_idx - start_idx) >= min_window_bars:
            break

    # If we still ended up too narrow (e.g. low was at/near the first bar and
    # recovered instantly), extend forward to guarantee a usable window
    if (end_idx - start_idx) < min_window_bars:
        end_idx = min(len(day_df) - 1, start_idx + min_window_bars)

    window = day_df.iloc[start_idx:end_idx + 1].copy()
    return window, low_price, low_time


def compute_volume_profile_poc(window_df, bin_size_pct=0.05):
    """Bin prices within the window and sum volume per bin to find the POC
    (highest-volume price node) and the 70% value area around it."""
    low = window_df["low"].min()
    high = window_df["high"].max()
    mid_price = (low + high) / 2
    bin_width = mid_price * (bin_size_pct / 100)
    if bin_width <= 0:
        bin_width = 0.05

    bins = {}
    for _, row in window_df.iterrows():
        # distribute this candle's volume across the price bins it spans
        candle_low, candle_high, vol = row["low"], row["high"], row["volume"]
        n_bins = max(1, int((candle_high - candle_low) / bin_width) + 1)
        vol_per_bin = vol / n_bins
        price = candle_low
        while price <= candle_high:
            bin_key = round(price / bin_width) * bin_width
            bins[bin_key] = bins.get(bin_key, 0) + vol_per_bin
            price += bin_width

    if not bins:
        return None

    sorted_bins = sorted(bins.items(), key=lambda x: x[0])
    total_vol = sum(v for _, v in sorted_bins)
    poc_price, poc_vol = max(sorted_bins, key=lambda x: x[1])

    # Build value area by expanding outward from POC until VALUE_AREA_PCT of volume covered
    poc_idx = [i for i, (p, v) in enumerate(sorted_bins) if p == poc_price][0]
    included = {poc_idx}
    covered_vol = poc_vol
    lo, hi = poc_idx, poc_idx
    while covered_vol < total_vol * VALUE_AREA_PCT and (lo > 0 or hi < len(sorted_bins) - 1):
        vol_below = sorted_bins[lo - 1][1] if lo > 0 else -1
        vol_above = sorted_bins[hi + 1][1] if hi < len(sorted_bins) - 1 else -1
        if vol_above >= vol_below:
            hi += 1
            covered_vol += sorted_bins[hi][1]
        else:
            lo -= 1
            covered_vol += sorted_bins[lo][1]

    val_low = sorted_bins[lo][0]
    val_high = sorted_bins[hi][0]

    return {
        "poc": round(poc_price, 2),
        "value_area_low": round(val_low, 2),
        "value_area_high": round(val_high, 2),
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_reversal_zone.py SYMBOL [DATE]")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    target_date = sys.argv[2] if len(sys.argv) > 2 else None

    if not ACCESS_TOKEN:
        print("ERROR: No Upstox access token found. Set it first:")
        print('  $env:UPSTOX_ACCESS_TOKEN = "your_token_here"')
        sys.exit(1)

    print(f"Resolving instrument key for {symbol} ...")
    instrument_key = resolve_equity_instrument_key(symbol)
    if not instrument_key:
        print(f"Could not find instrument key for {symbol}")
        sys.exit(1)

    print("Fetching 5-minute candles ...")
    if target_date:
        target_dt = pd.to_datetime(target_date)
        to_date = (target_dt + timedelta(days=5)).strftime("%Y-%m-%d")
        from_date = (target_dt - timedelta(days=5)).strftime("%Y-%m-%d")
        df = fetch_candles(instrument_key, to_date=to_date, from_date=from_date)
    else:
        df = fetch_candles(instrument_key)

    if df.empty:
        print("No candle data returned.")
        sys.exit(1)

    if target_date:
        day_df = df[df["date"] == pd.to_datetime(target_date).date()]
        if day_df.empty:
            print(f"No data found for {target_date}. Available dates: {sorted(df['date'].unique())}")
            sys.exit(1)
    else:
        latest_date = df["date"].max()
        day_df = df[df["date"] == latest_date]
        print(f"No date specified, using most recent trading day: {latest_date}")

    window, low_price, low_time = find_swing_low_window(day_df)
    print(f"\nSwing low: {low_price} at {low_time.strftime('%Y-%m-%d %H:%M')}")
    print(f"Analyzing reversal window: {window['timestamp'].iloc[0].strftime('%H:%M')} "
          f"to {window['timestamp'].iloc[-1].strftime('%H:%M')} ({len(window)} candles)")

    profile = compute_volume_profile_poc(window)
    if profile is None:
        print("Could not compute volume profile (insufficient data).")
        sys.exit(1)

    print(f"\n=== Reversal Support Zone (volume profile of the basing period) ===")
    print(f"POC (highest volume node): {profile['poc']}")
    print(f"Value Area: {profile['value_area_low']} - {profile['value_area_high']}")
    print(f"\nThis is where real buying absorption happened after the breakdown - "
          f"a more precise support reference than the original (broken) Delta Support level.")


if __name__ == "__main__":
    main()
