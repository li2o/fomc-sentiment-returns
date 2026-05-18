#!/usr/bin/env python3
"""
Bitcoin Data API Fetcher

Fetches Bitcoin OHLC data from public APIs and stores it locally.
CoinGecko API: https://docs.coingecko.com/
Bitstamp API: https://www.bitstamp.net/api/
"""

import csv
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import requests
import pandas as pd

# CoinGecko API configuration
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
BITCOIN_ID = "bitcoin"
DEFAULT_CURRENCY = "usd"

# File paths
DEFAULT_OUTPUT_FILE = "bitcoin_hourly_data.csv"
DEFAULT_JSON_FILE = "bitcoin_hourly_data.json"

# Bitstamp API configuration (exchange-specific BTC/USD)
BITSTAMP_BASE = "https://www.bitstamp.net/api/v2"
BITSTAMP_PAIR = "btcusd"
BITSTAMP_STEP_1H = 3600
BITSTAMP_MAX_LIMIT = 1000

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between requests
BITSTAMP_REQUEST_DELAY = 0.25

# Headers to avoid blocking
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    )
}


def _to_unix_ts(dt: datetime) -> int:
    return int(dt.timestamp())


def fetch_bitstamp_ohlc_chunk(
    start_ts: int,
    end_ts: int,
    step_sec: int = BITSTAMP_STEP_1H,
    limit: int = BITSTAMP_MAX_LIMIT,
    retry_count: int = 3
) -> Optional[List[Dict[str, str]]]:
    """
    Fetch a single OHLC chunk from Bitstamp.

    Args:
        start_ts: Unix timestamp (seconds)
        end_ts: Unix timestamp (seconds)
        step_sec: Candle size in seconds (e.g., 1800 for 30 minutes)
        limit: Max candles per request (Bitstamp allows up to 1000)
        retry_count: Retries on failure

    Returns:
        List of OHLC dicts or None on failure
    """

    url = f"{BITSTAMP_BASE}/ohlc/{BITSTAMP_PAIR}/"
    params = {
        "step": step_sec,
        "limit": limit,
        "start": start_ts,
        "end": end_ts,
    }

    for attempt in range(retry_count):
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", {}).get("ohlc", [])
            return data
        except requests.exceptions.RequestException as e:
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2
                print(f"Bitstamp request failed ({str(e)[:60]}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch Bitstamp data after {retry_count} attempts: {e}")
                return None


def fetch_bitstamp_ohlc_series(
    start_dt: datetime,
    end_dt: datetime,
    step_sec: int = BITSTAMP_STEP_1H,
    limit: int = BITSTAMP_MAX_LIMIT,
    sleep_s: float = BITSTAMP_REQUEST_DELAY
) -> List[Dict[str, str]]:
    """
    Fetch Bitstamp OHLC candles over a full time range.

    Args:
        start_dt: Start datetime (UTC recommended)
        end_dt: End datetime (UTC recommended)
        step_sec: Candle size in seconds
        limit: Max candles per request
        sleep_s: Delay between requests

    Returns:
        List of OHLC dicts across the full time range
    """

    start_ts = _to_unix_ts(start_dt)
    end_ts = _to_unix_ts(end_dt)
    chunk_span = step_sec * limit

    all_rows: List[Dict[str, str]] = []
    seen_ts = set()
    cursor = start_ts

    while cursor < end_ts:
        chunk_end = min(cursor + chunk_span, end_ts)
        print(f"Fetching Bitstamp OHLC: {cursor} -> {chunk_end}")
        rows = fetch_bitstamp_ohlc_chunk(cursor, chunk_end, step_sec=step_sec, limit=limit)
        if rows is None:
            break

        for row in rows:
            ts = int(row.get("timestamp", "0"))
            if ts == 0 or ts in seen_ts:
                continue
            seen_ts.add(ts)
            all_rows.append(row)

        cursor = chunk_end
        time.sleep(max(0.0, sleep_s))

    return all_rows


def parse_bitstamp_to_dataframe(ohlc_rows: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Convert Bitstamp OHLC rows to a pandas DataFrame.

    Bitstamp returns dicts with string values:
    {"timestamp", "open", "high", "low", "close", "volume"}
    """

    rows = []
    for row in ohlc_rows:
        try:
            ts = int(row["timestamp"])
            rows.append({
                "timestamp": datetime.utcfromtimestamp(ts),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            })
        except (KeyError, ValueError, TypeError):
            continue

    df = pd.DataFrame(rows)
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_and_save_bitstamp_data(
    start_dt: datetime,
    end_dt: datetime,
    step_sec: int = BITSTAMP_STEP_1H,
    output_file: str = "bitcoin_bitstamp_1h.csv",
    output_json: Optional[str] = "bitcoin_bitstamp_1h.json"
) -> Optional[pd.DataFrame]:
    """
    Fetch Bitstamp BTC/USD OHLC data and save to CSV/JSON.

    Args:
        start_dt: Start datetime (UTC recommended)
        end_dt: End datetime (UTC recommended)
        step_sec: Candle size in seconds (3600 for 1 hour)
        output_file: CSV output path
        output_json: JSON output path (None to skip)
    """

    rows = fetch_bitstamp_ohlc_series(start_dt, end_dt, step_sec=step_sec)
    if not rows:
        print("No Bitstamp data returned")
        return None

    df = parse_bitstamp_to_dataframe(rows)
    if df.empty:
        print("No Bitstamp data to process")
        return None

    print(f"Processed {len(df)} Bitstamp candles")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    first_ts = df["timestamp"].min()
    print(f"First timestamp (UTC): {first_ts.isoformat()}")
    print(f"First timestamp (unix): {int(first_ts.timestamp())}")

    save_to_csv(df, output_file)
    if output_json:
        save_to_json(df, output_json)

    return df


def fetch_ohlcv_data(
    days: int = 30,
    vs_currency: str = DEFAULT_CURRENCY,
    retry_count: int = 3,
    interval: Optional[str] = None
) -> Optional[List[List]]:
    """
    Fetch OHLCV (Open, High, Low, Close, Volume) data for Bitcoin from CoinGecko.
    
    Granularity information:
    - 1-2 days: automatic 30-minute data
    - 3-30 days: automatic 4-hour data
    - 31+ days: automatic 4-hour data (up to 90 days)
    - 'max': all historical data from 2013 with 4-day candles
    
    Note: Interval parameter (hourly/daily) only works with paid API keys
    
    Args:
        days: Number of days to fetch. Options: 1, 7, 14, 30, 90, 180, 365, 'max'
              or integer for auto granularity
        vs_currency: Currency to fetch prices in (default: 'usd')
        retry_count: Number of retries on failure
        interval: Optional 'hourly' or 'daily' (requires paid API key)
    
    Returns:
        List of [timestamp, open, high, low, close, volume] lists, or None on failure
    """
    
    url = f"{COINGECKO_BASE}/coins/{BITCOIN_ID}/ohlc"
    params = {
        "vs_currency": vs_currency,
    }
    
    # Handle days parameter
    if isinstance(days, int):
        if days > 90:
            print(f"Warning: Free tier supports max 90 days for hourly/daily intervals.")
            print(f"Using 'max' for full historical data (4-day granularity after 30 days).")
            params["days"] = "max"
        else:
            params["days"] = days
    else:
        params["days"] = days  # 'max' string
    
    if interval:
        params["interval"] = interval
    
    for attempt in range(retry_count):
        try:
            print(f"Fetching Bitcoin OHLCV data (days={params.get('days')}, attempt {attempt + 1}/{retry_count})...")
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            print(f"Successfully fetched {len(data)} OHLCV candles")
            
            # Print granularity info
            if len(data) > 1:
                ts1 = data[0][0]
                ts2 = data[1][0]
                minute_diff = (ts2 - ts1) / (1000 * 60)
                granularity = f"{int(minute_diff)} minutes" if minute_diff < 60 else f"{int(minute_diff / 60)} hours"
                print(f"Data granularity: {granularity}")
            
            return data
            
        except requests.exceptions.RequestException as e:
            if attempt < retry_count - 1:
                wait_time = (attempt + 1) * 2
                print(f"Request failed ({str(e)[:50]}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to fetch data after {retry_count} attempts: {e}")
                return None


def fetch_market_chart_data(
    days: int = 365,
    vs_currency: str = DEFAULT_CURRENCY
) -> Optional[Dict]:
    """
    Fetch detailed market chart data including prices, market caps, and volumes.
    Note: This provides daily data, not hourly. For hourly, use fetch_ohlcv_data.
    
    Args:
        days: Number of days of historical data
        vs_currency: Currency to fetch data in
    
    Returns:
        Dictionary with 'prices', 'market_caps', 'volumes' or None on failure
    """
    
    url = f"{COINGECKO_BASE}/coins/{BITCOIN_ID}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily"
    }
    
    try:
        print(f"Fetching Bitcoin market chart data for {days} days...")
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        print(f"Successfully fetched market chart data")
        return data
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch market chart data: {e}")
        return None


def parse_ohlcv_to_dataframe(ohlcv_data: List[List]) -> pd.DataFrame:
    """
    Convert OHLCV data from CoinGecko into a pandas DataFrame.
    
    CoinGecko returns [timestamp_ms, open, high, low, close, volume]
    
    Args:
        ohlcv_data: Raw OHLCV data from CoinGecko API
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    
    rows = []
    for candle in ohlcv_data:
        if len(candle) < 5:
            continue
        
        timestamp_ms = candle[0]
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000)
        
        rows.append({
            'timestamp': timestamp,
            'open': candle[1],
            'high': candle[2],
            'low': candle[3],
            'close': candle[4],
            'volume': candle[5] if len(candle) > 5 else None,
        })
    
    df = pd.DataFrame(rows)
    return df.sort_values('timestamp').reset_index(drop=True)


def save_to_csv(df: pd.DataFrame, filepath: str = DEFAULT_OUTPUT_FILE) -> bool:
    """
    Save Bitcoin data to CSV file.
    
    Args:
        df: DataFrame with Bitcoin data
        filepath: Output CSV file path
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} records to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save CSV: {e}")
        return False


def save_to_json(df: pd.DataFrame, filepath: str = DEFAULT_JSON_FILE) -> bool:
    """
    Save Bitcoin data to JSON file.
    
    Args:
        df: DataFrame with Bitcoin data
        filepath: Output JSON file path
    
    Returns:
        True if successful, False otherwise
    """
    
    try:
        # Convert DataFrame to JSON with orient='records' for list of objects
        df_copy = df.copy()
        df_copy['timestamp'] = df_copy['timestamp'].astype(str)
        
        with open(filepath, 'w') as f:
            json.dump(df_copy.to_dict(orient='records'), f, indent=2)
        
        print(f"Saved {len(df)} records to {filepath}")
        return True
    except Exception as e:
        print(f"Failed to save JSON: {e}")
        return False


def load_existing_data(filepath: str = DEFAULT_OUTPUT_FILE) -> Optional[pd.DataFrame]:
    """
    Load existing Bitcoin data from CSV file.
    
    Args:
        filepath: Input CSV file path
    
    Returns:
        DataFrame or None if file doesn't exist
    """
    
    if not os.path.exists(filepath):
        return None
    
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Loaded {len(df)} existing records from {filepath}")
        return df
    except Exception as e:
        print(f"Failed to load existing data: {e}")
        return None


def fetch_and_save_bitcoin_data(
    days: int = 30,
    vs_currency: str = DEFAULT_CURRENCY,
    output_file: str = DEFAULT_OUTPUT_FILE,
    output_json: Optional[str] = DEFAULT_JSON_FILE,
    append: bool = False,
    interval: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Main function: Fetch Bitcoin OHLCV data and save to files.
    
    Args:
        days: Number of days of historical data. Options: 1, 7, 14, 30, 90, 180, 365, 'max'
        vs_currency: Currency for prices
        output_file: CSV output file path
        output_json: JSON output file path (None to skip)
        append: If True, append to existing file; if False, overwrite
        interval: Optional 'hourly' or 'daily' (requires paid API key)
    
    Returns:
        DataFrame with fetched data, or None on failure
    """
    
    # Fetch OHLCV data
    ohlcv_raw = fetch_ohlcv_data(days=days, vs_currency=vs_currency, interval=interval)
    if ohlcv_raw is None:
        return None
    
    # Convert to DataFrame
    df_new = parse_ohlcv_to_dataframe(ohlcv_raw)
    if df_new.empty:
        print("No data to process")
        return None
    
    print(f"Processed {len(df_new)} OHLCV candles")
    print(f"Date range: {df_new['timestamp'].min()} to {df_new['timestamp'].max()}")
    
    # Append to existing data if requested
    if append:
        df_existing = load_existing_data(output_file)
        if df_existing is not None:
            # Combine and remove duplicates based on timestamp
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
            df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)
            df_final = df_combined
            print(f"Combined with existing data: {len(df_final)} total records")
        else:
            df_final = df_new
    else:
        df_final = df_new
    
    # Save outputs
    save_to_csv(df_final, output_file)
    if output_json:
        save_to_json(df_final, output_json)
    
    return df_final


def fetch_all_historical_bitcoin_data(
    vs_currency: str = DEFAULT_CURRENCY,
    output_file: str = DEFAULT_OUTPUT_FILE,
    output_json: Optional[str] = DEFAULT_JSON_FILE
) -> Optional[pd.DataFrame]:
    """
    Fetch ALL available historical Bitcoin data from 2013 to today.
    
    IMPORTANT NOTES:
    - Bitcoin data only available from ~January 2013 onwards (not from 2011)
    - For periods beyond 30 days, CoinGecko provides 4-hour candles (not 30-minute)
    - Full dataset will contain ~26,000-30,000 candles
    - Use this for complete historical analysis
    
    Args:
        vs_currency: Currency for prices
        output_file: CSV output file path
        output_json: JSON output file path  
    
    Returns:
        DataFrame with all historical data, or None on failure
    """
    
    print("="*70)
    print("FETCHING ALL HISTORICAL BITCOIN DATA (2013-TODAY)")
    print("="*70)
    print("\nIMPORTANT INFORMATION:")
    print("- Bitcoin data starts from January 2013 (not 2011)")
    print("- Data granularity:")
    print("  * First 2 days: 30-minute candles")
    print("  * Days 3-30: 4-hour candles")
    print("  * Beyond 30 days: 4-hour candles")
    print("- This will fetch approximately 26,000-30,000 candles")
    print("="*70 + "\n")
    
    return fetch_and_save_bitcoin_data(
        days='max',
        vs_currency=vs_currency,
        output_file=output_file,
        output_json=output_json,
        append=False
    )


def get_latest_bitcoin_price(vs_currency: str = DEFAULT_CURRENCY) -> Optional[Dict]:
    """
    Fetch the latest Bitcoin price and market data.
    
    Args:
        vs_currency: Currency for price
    
    Returns:
        Dictionary with price data or None on failure
    """
    
    url = f"{COINGECKO_BASE}/coins/{BITCOIN_ID}"
    params = {"localization": "false"}
    
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        market_data = data.get('market_data', {})
        
        return {
            'price': market_data.get('current_price', {}).get(vs_currency),
            'market_cap': market_data.get('market_cap', {}).get(vs_currency),
            '24h_change': market_data.get('price_change_percentage_24h'),
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Failed to fetch latest price: {e}")
        return None


def print_data_summary(df: pd.DataFrame) -> None:
    """Print summary statistics of Bitcoin data."""
    
    if df.empty:
        print("No data to summarize")
        return
    
    print("\n" + "="*60)
    print("Bitcoin Data Summary")
    print("="*60)
    print(f"Records: {len(df)}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nPrice Statistics (Close):")
    print(f"  Min:  ${df['close'].min():.2f}")
    print(f"  Max:  ${df['close'].max():.2f}")
    print(f"  Mean: ${df['close'].mean():.2f}")
    print(f"  Latest: ${df['close'].iloc[-1]:.2f}")
    
    if 'volume' in df.columns:
        print(f"\nVolume Statistics:")
        print(f"  Min:  {df['volume'].min():.2f}")
        print(f"  Max:  {df['volume'].max():.2f}")
        print(f"  Mean: {df['volume'].mean():.2f}")
    print("="*60 + "\n")


def main():
    """Example usage of the Bitcoin data fetcher."""
    
    print("Bitcoin Data API Fetcher")
    print("-" * 60)
    
    # OPTION 0 (DEFAULT): Bitstamp 1-hour BTC/USD candles from 2011 to today
    # Note: availability depends on Bitstamp's historical coverage for BTC/USD.
    print("\nOption 0: Fetching Bitstamp 1-hour BTC/USD data (2011-today)...\n")
    df = fetch_and_save_bitstamp_data(
        start_dt=datetime(2011, 1, 1),
        end_dt=datetime.utcnow(),
        step_sec=BITSTAMP_STEP_1H,
        output_file="bitcoin_bitstamp_1h.csv",
        output_json="bitcoin_bitstamp_1h.json"
    )
    
    if df is not None:
        print_data_summary(df)
    
    # OPTION 1: Fetch all historical data from 2013 to today (CoinGecko)
    # This will get 4-hour candles for long periods, 30-minute for last 2 days
    # print("\nOption 1: Fetching ALL historical Bitcoin data (2013-today)...\n")
    # df = fetch_all_historical_bitcoin_data(
    #     vs_currency='usd',
    #     output_file='bitcoin_all_historical.csv',
    #     output_json='bitcoin_all_historical.json'
    # )
    # if df is not None:
    #     print_data_summary(df)

    # OPTION 2: Fetch recent 30-day data (4-hour granularity)
    # Uncomment to use instead:
    # print("\nOption 2: Fetching last 30 days of Bitcoin data...\n")
    # df = fetch_and_save_bitcoin_data(
    #     days=30,
    #     vs_currency='usd',
    #     output_file='bitcoin_30day.csv',
    #     output_json='bitcoin_30day.json',
    #     append=False
    # )
    # if df is not None:
    #     print_data_summary(df)
    
    # OPTION 3: Fetch recent data with 30-minute granularity (max 2 days)
    # Uncomment to use instead:
    # print("\nOption 3: Fetching last 2 days with 30-minute granularity...\n")
    # df = fetch_and_save_bitcoin_data(
    #     days=2,
    #     vs_currency='usd',
    #     output_file='bitcoin_30min.csv',
    #     output_json='bitcoin_30min.json',
    #     append=False
    # )
    # if df is not None:
    #     print_data_summary(df)
    
    # Get latest price
    if df is not None:
        print("\nFetching latest Bitcoin price...")
        latest = get_latest_bitcoin_price()
        if latest:
            print(f"Current BTC Price: ${latest['price']:.2f} USD")
            print(f"24h Change: {latest['24h_change']:.2f}%")
            print(f"Market Cap: ${latest['market_cap']:,.0f} USD")


if __name__ == "__main__":
    main()
