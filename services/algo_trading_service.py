"""
Algo Trading Service Module
A FastAPI router module for algo trading functionality.
"""
from fastapi import APIRouter, HTTPException
import asyncio
import numpy as np
import pandas as pd
import requests
import holidays
import time as time_module
from datetime import datetime, timedelta, time
from zoneinfo import ZoneInfo
import random
from io import StringIO
from typing import List, Optional, Dict, Any
# import threading  # removed: no longer needed after migrating from ibapi to ib_async
from sqlalchemy import create_engine, text
import psycopg
from psycopg.rows import dict_row
from sqlalchemy.engine import make_url
import ollama
import os, sys

from shared.config import BaseServiceConfig

from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

from dataclasses import dataclass

### IBKR (ib_async) — pip install ib_async (https://github.com/ib-api-reloaded/ib_async)
from ib_async import IB, Stock, MarketOrder, LimitOrder, Contract, Order, util


@dataclass
class PairSignal:
    signal: int               # +1 long spread, -1 short spread, 0 flat
    zscore: float
    beta: float
    gross_dollars: float
    tom_dollars: float
    jerry_dollars: float
    last_price_tom: float
    last_price_jerry: float
    asof: pd.Timestamp

# Create router for this service
router = APIRouter(prefix="/algo_trading", tags=["algo_trading"])

config = BaseServiceConfig()
db_url = f"postgresql://{config.postgres_user_algo_trading}:{config.postgres_password_algo_trading}@{config.postgres_host_algo_trading}:{config.postgres_port_algo_trading}/{config.postgres_db_algo_trading}"

# Convert SQLAlchemy-style URL into psycopg connection string
# psycopg_url = make_url(db_url)
# psycopg_conninfo = (
#     f"host={psycopg_url.host} port={psycopg_url.port} dbname={psycopg_url.database} "
#     f"user={psycopg_url.username} password={psycopg_url.password}"
# )

psycopg_conninfo = (
    f"host={config.postgres_host_algo_trading} port={config.postgres_port_algo_trading} dbname={config.postgres_db_algo_trading} "
    f"user={config.postgres_user_algo_trading} password={config.postgres_password_algo_trading}"
)

engine = create_engine(db_url.replace("postgresql://", "postgresql+psycopg://"))
scaler = StandardScaler()


## Open https://stooq.com/q/d/?s=aapl.us&get_apikey

def signal_to_side(signal_value: int) -> str:
    if signal_value == 1:
        return "LONG_SPREAD"
    if signal_value == -1:
        return "SHORT_SPREAD"
    return "NONE"


def save_trade_candidates_df(
    candidates_df: pd.DataFrame,
    trade_date: Optional[pd.Timestamp] = None,
    table_name: str = "trade_candidates",
) -> None:
    if candidates_df.empty:
        return

    df_db = candidates_df.copy()
    if trade_date is None:
        trade_date = pd.Timestamp.now(tz="America/New_York").date()

    df_db['trade_date'] = trade_date
    df_db = df_db.rename(columns={
        "pair_tom": "pair_symbol_1",
        "pair_jerry": "pair_symbol_2"
        })

    log(f"save_trade_candidates_df()::df_db \n{df_db}")

    # Build CSV buffer from dataframe
    columns = [
        "trade_date",
        "pair_symbol_1",
        "pair_symbol_2",
        "signal",
        "zscore",
        "abs_z",
        "gross_dollars",
    ]

    csv_buffer = StringIO()
    df_db[columns].to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    create_staging_sql = f"""
        CREATE TEMP TABLE staging_{table_name} (LIKE {table_name} INCLUDING DEFAULTS)
        ON COMMIT DROP;
    """

    copy_sql = f"""
        COPY staging_{table_name} (
            trade_date, pair_symbol_1, pair_symbol_2,
            signal, zscore, abs_z, gross_dollars
        )
        FROM STDIN WITH (FORMAT CSV);
    """

    upsert_sql = f"""
        INSERT INTO {table_name} (
            trade_date, pair_symbol_1, pair_symbol_2,
            signal, zscore, abs_z, gross_dollars
        )
        SELECT
            trade_date, pair_symbol_1, pair_symbol_2,
            signal, zscore, abs_z, gross_dollars
        FROM staging_{table_name}
        ON CONFLICT (trade_date, pair_symbol_1, pair_symbol_2)
        DO UPDATE SET
            signal        = EXCLUDED.signal,
            zscore        = EXCLUDED.zscore,
            abs_z         = EXCLUDED.abs_z,
            gross_dollars = EXCLUDED.gross_dollars,
            created_at    = NOW();
    """

    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(create_staging_sql)

            with cur.copy(copy_sql) as copy:
                copy.write(csv_buffer.read())

            cur.execute(upsert_sql)

        conn.commit()


def create_trade_record(
    pair_tom: str,
    pair_jerry: str,
    signal: "PairSignal",
    trade_plan: Dict[str, Any],
    strategy_name: str = "pairs_trading",
    risk_amount: Optional[float] = None,
    notes: Optional[str] = None,
) -> Dict[str, Any]:
    leg_tom = trade_plan["leg_tom"]
    leg_jerry = trade_plan["leg_jerry"]

    insert_sql = """
    INSERT INTO trades (
        strategy_name,
        pair_symbol_1,
        pair_symbol_2,
        side,
        entry_zscore,
        entry_spread,
        signal_time,
        qty_1,
        qty_2,
        hedge_ratio,
        entry_price_1,
        entry_price_2,
        status,
        risk_amount,
        notes
    ) VALUES (
        :strategy_name,
        :pair_symbol_1,
        :pair_symbol_2,
        :side,
        :entry_zscore,
        :entry_spread,
        :signal_time,
        :qty_1,
        :qty_2,
        :hedge_ratio,
        :entry_price_1,
        :entry_price_2,
        :status,
        :risk_amount,
        :notes
    )
    RETURNING id;
    """

    params = {
        "strategy_name": strategy_name,
        "pair_symbol_1": pair_tom,
        "pair_symbol_2": pair_jerry,
        "side": signal_to_side(signal.signal),
        "entry_zscore": float(signal.zscore),
        "entry_spread": None,
        "signal_time": pd.Timestamp(signal.asof).to_pydatetime(),
        "qty_1": int(leg_tom["quantity"]),
        "qty_2": int(leg_jerry["quantity"]),
        "hedge_ratio": float(signal.beta),
        "entry_price_1": float(signal.last_price_tom),
        "entry_price_2": float(signal.last_price_jerry),
        "status": "PENDING",
        "risk_amount": float(risk_amount) if risk_amount is not None else None,
        "notes": notes,
    }

    with engine.begin() as conn:
        trade_id = conn.execute(text(insert_sql), params).scalar_one()
        order_ref = f"trade:{trade_id}"
        conn.execute(
            text("UPDATE trades SET order_ref = :order_ref, updated_at = NOW() WHERE id = :trade_id"),
            {"order_ref": order_ref, "trade_id": trade_id},
        )

    return {"trade_id": int(trade_id), "order_ref": order_ref}


def update_trade_after_order_submission(trade_id: int, order_results: List[Dict[str, Any]]) -> None:
    order_ids = [o.get("order_id") for o in order_results if o.get("submitted")]
    if not order_ids:
        return

    first_order_id = order_ids[0] if len(order_ids) > 0 else None
    second_order_id = order_ids[1] if len(order_ids) > 1 else None
    new_status = "OPEN" if len(order_ids) == 2 else "PENDING"

    update_sql = """
    UPDATE trades
    SET
        ib_order_id_1 = COALESCE(:ib_order_id_1, ib_order_id_1),
        ib_order_id_2 = COALESCE(:ib_order_id_2, ib_order_id_2),
        entry_time = CASE WHEN :mark_open THEN NOW() ELSE entry_time END,
        status = :status,
        updated_at = NOW()
    WHERE id = :trade_id;
    """

    with engine.begin() as conn:
        conn.execute(
            text(update_sql),
            {
                "trade_id": trade_id,
                "ib_order_id_1": first_order_id,
                "ib_order_id_2": second_order_id,
                "mark_open": len(order_ids) == 2,
                "status": new_status,
            },
        )


def mark_candidate_promoted(
    trade_date: pd.Timestamp,
    pair_tom: str,
    pair_jerry: str,
    trade_id: int,
    table_name: str = "trade_candidates",
) -> None:
    update_sql = f"""
    UPDATE {table_name}
    SET
        promoted_to_trade = TRUE,
        trade_id = :trade_id
    WHERE trade_date = :trade_date
      AND pair_symbol_1 = :pair_symbol_1
      AND pair_symbol_2 = :pair_symbol_2;
    """

    with engine.begin() as conn:
        conn.execute(
            text(update_sql),
            {
                "trade_id": trade_id,
                "trade_date": pd.to_datetime(trade_date).date(),
                "pair_symbol_1": pair_tom,
                "pair_symbol_2": pair_jerry,
            },
        )

# used by _calculate_cluster_and_cointegration
cluster_dict = {}

def log(*args,
        sep=" ",
        end="\n",
        file=None,
        flush=False):
    """
    Drop-in replacement for print() with timestamp prefix.
    """

    if file is None:
        file = sys.stdout

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    message = sep.join(str(arg) for arg in args)

    print(f"[{now}] {message}",
          sep=sep,
          end=end,
          file=file,
          flush=flush)

@router.get("/health")
async def health_check():
    """Health check endpoint for algo trading service."""
    return {
        "service": "algo_trading",
        "status": "healthy"
    }

def clean_nasdaqtrader_df(df_input):
    df_copy = df_input.copy()

    # remove non-equity / preferred / others
    keywords = [
        "Note", "Notes", "Bond", "Debenture",
        "Preferred", "Depositary",
        "Warrant", "Warrants", "Unit", "Units",
        "Right", "Rights"
    ]
    pattern = r'\b(?:' + '|'.join(keywords) + r')\b'
    # pattern = r'\b(?:Note|Bond|Debenture|Preferred|Depositary|Warrant|Unit|Units|Right|Rights)\b'
    df_copy = df_copy[~df_copy['Security Name'].str.contains(pattern, case=False, na=False)]
    
    if "ACT Symbol" in df_copy.columns:
        df_copy = df_copy.rename(columns={"ACT Symbol": "Symbol"})
    
    if 'Symbol' in df_copy.columns: 
        # drop row starting with "File Creation Time:", usually last row of newly downloaded file
        df_copy = df_copy[
            ~df_copy['Symbol'].str.strip().str.startswith('File Creation Time:', na=False)
        ]
    
    if 'Financial Status' in df_copy.columns:
        df_copy = df_copy[df_copy['Financial Status'] == 'N']

    if 'Test Issue' in df_copy.columns:
        df_copy = df_copy[df_copy['Test Issue'] == 'N']

    if 'Exchange' in df_copy.columns:
        df_copy = df_copy[df_copy['Exchange'] == 'N']

    if 'ETF' in df_copy.columns:
        df_copy = df_copy[df_copy['ETF'] == 'N']

    # clean up security name
    df_copy['Security Name'] = (df_copy['Security Name']
        .str.strip()
        .str.split(' - ').str[0]
        .str.replace(r'\s*-\s*.*$|\s+(Class [A-Z]\s+)?(Common Stock|Common Shares|Ordinary Shares)$', '', regex=True)
              )
    return df_copy

def fetch_stooq_symbol(
    symbol: str,
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data for one symbol from Stooq.

    Parameters
    ----------
    symbol : str
        Example: 'AAPL.US'
    interval : str
        'd' = daily, 'w' = weekly, 'm' = monthly
    start : str | None
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD'
    end : str | None
        End date in 'YYYYMMDD' or 'YYYY-MM-DD'
    session : requests.Session | None
        Optional shared session for performance

    Returns
    -------
    pd.DataFrame
        Columns:
        symbol, date, open, high, low, close, volume
    """

    def format_date(d: str) -> str:
        # Accept 'YYYY-MM-DD' or 'YYYYMMDD'
        return pd.to_datetime(d).strftime("%Y%m%d")

    symbol_lower = symbol.lower()
    if not symbol_lower.endswith(".us"):
        symbol_lower = symbol_lower + ".us"

    # Build URL dynamically
    apikey = "yVD4FuTlhwvckyxnq80sQA7jeg3EmM9L" # None

    url = f"https://stooq.com/q/d/l/?s={symbol_lower}&i={interval}"
    if start:
        url += f"&d1={format_date(start)}"
    if end:
        url += f"&d2={format_date(end)}"
    if apikey:
        url += f"&apikey={apikey}"

    log(f"fetch_stooq_symbol()::url: {url}")
    sess = session or requests.Session()
    response = sess.get(url, timeout=20)
    response.raise_for_status()

    text = response.text.strip()
    if not text or "No data" in text:
        raise ValueError(f"No data returned for symbol: {symbol}")

    df = pd.read_csv(StringIO(text))

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"date", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    df["symbol"] = symbol_lower
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove bad rows
    df = df.dropna(subset=["date"])

    # Sort ascending (Stooq returns newest first)
    df = df.sort_values("date").reset_index(drop=True)
    df["source"] = "stooq"
    df["timeframe"] = "d"

    # Reorder columns
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "source", "timeframe"]]

    return df

def fetch_yfinance_symbol(
    symbol: str,
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download OHLCV data for one symbol from Yahoo Finance via yfinance.

    Parameters
    ----------
    symbol : str
        Ticker symbol, e.g. 'AAPL' or 'AAPL.US' (the .US suffix is stripped automatically)
    interval : str
        'd' = daily, 'w' = weekly, 'm' = monthly
    start : str | None
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD' (inclusive)
    end : str | None
        End date in 'YYYYMMDD' or 'YYYY-MM-DD' (inclusive)

    Returns
    -------
    pd.DataFrame
        Columns: symbol, date, open, high, low, close, volume, source, timeframe
    """
    import yfinance as yf

    interval_map = {"d": "1d", "w": "1wk", "m": "1mo"}
    yf_interval = interval_map.get(interval, "1d")

    symbol_clean = symbol.upper()
    if symbol_clean.endswith(".US"):
        symbol_clean = symbol_clean[:-3]

    # yfinance end date is exclusive; add 1 day to match stooq inclusive behaviour
    end_dt = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
    start_dt = pd.to_datetime(start).strftime("%Y-%m-%d") if start else None

    log(f"fetch_yfinance_symbol()::symbol={symbol_clean} interval={yf_interval} start={start_dt} end={end_dt}")

    df = yf.Ticker(symbol_clean).history(
        start=start_dt,
        end=end_dt,
        interval=yf_interval,
        auto_adjust=True,
        actions=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for symbol: {symbol}")

    df = df.reset_index()
    df.columns = [c.strip().lower() for c in df.columns]

    # Daily history returns 'date', intraday returns 'datetime'
    if "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = symbol_clean
    df["source"] = "yfinance"
    df["timeframe"] = interval

    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "source", "timeframe"]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def bulk_download_stooq(
    symbols: List[str],
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    save: bool = False,
    sleep_min: float = 0.4,
    sleep_max: float = 1.2,
) -> pd.DataFrame:
    """
    Download many symbols from Stooq with light throttling.

    Parameters
    ----------
    symbols : list[str]
        Example: ['AAPL.US', 'MSFT.US', 'GOOG.US']
    interval : str
        d / w / m
    start : str | None
        Start date (YYYYMMDD or YYYY-MM-DD)
    end : str | None
        End date (YYYYMMDD or YYYY-MM-DD)
    save : bool
        Save to database
    sleep_min, sleep_max : float
        Random delay between requests to reduce risk of blocking

    Returns
    -------
    pd.DataFrame
        Combined OHLCV dataframe for all symbols
    """

    all_frames = []
    batch_frames = []  # <-- for every 10 symbols
    failures = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/145.0.0.0 Safari/537.36"
                )
            }
        )

        for i, symbol in enumerate(symbols, start=1):
            try:
                df = fetch_stooq_symbol(
                    symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    session=session
                )

                all_frames.append(df)
                batch_frames.append(df)

                log(f"[{i}/{len(symbols)}] OK   {symbol}  rows={len(df)}")

                # 🔥 Save every 10 symbols
                if save and i % 10 == 0:
                    batch_df = pd.concat(batch_frames, ignore_index=True)
                    # save_ohlcv_outputs(batch_df)
                    save_ohlcv_outputs_fast(batch_df)
                    log(f"💾 Saved batch of 10 symbols (up to {symbol})")
                    batch_frames = []  # reset batch

            except Exception as e:
                failures.append((symbol, str(e)))
                log(f"[{i}/{len(symbols)}] FAIL {symbol}  error={e}")

            time_module.sleep(random.uniform(sleep_min, sleep_max))

    # 🔥 Save remaining symbols (<10)
    if save and batch_frames:
        batch_df = pd.concat(batch_frames, ignore_index=True)
        # save_ohlcv_outputs(batch_df)
        save_ohlcv_outputs_fast(batch_df)
        log(f"💾 Saved final batch of {len(batch_frames)} symbols")

    if not all_frames:
        raise RuntimeError("No symbols downloaded successfully.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates if any
    combined = combined.drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)

    if failures:
        log("\nFailures:")
        for sym, err in failures:
            log(f" - {sym}: {err}")

    return combined

def bulk_download_yfinance(
    symbols: List[str],
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    save: bool = False,
    batch_size: int = 100,
) -> pd.DataFrame:
    """
    Download OHLCV data for many symbols from Yahoo Finance in batches.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols, e.g. ['AAPL', 'MSFT'] or ['AAPL.US', 'MSFT.US']
    interval : str
        'd' = daily, 'w' = weekly, 'm' = monthly
    start : str | None
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD' (inclusive)
    end : str | None
        End date in 'YYYYMMDD' or 'YYYY-MM-DD' (inclusive)
    save : bool
        Persist each batch to the database as it downloads
    batch_size : int
        Number of symbols per yfinance API call (default 100)

    Returns
    -------
    pd.DataFrame
        Combined OHLCV dataframe: symbol, date, open, high, low, close, volume, source, timeframe
    """
    import yfinance as yf

    interval_map = {"d": "1d", "w": "1wk", "m": "1mo"}
    yf_interval = interval_map.get(interval, "1d")

    clean_symbols = [s.upper().removesuffix(".US") for s in symbols]

    start_dt = pd.to_datetime(start).strftime("%Y-%m-%d") if start else None
    # yfinance end is exclusive; add 1 day to match inclusive behaviour
    end_dt = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None

    all_frames = []
    failures = []

    def _normalise(sub: pd.DataFrame, sym: str) -> pd.DataFrame:
        sub = sub.dropna(how="all").reset_index()
        sub.columns = [c.strip().lower() for c in sub.columns]
        if "datetime" in sub.columns:
            sub = sub.rename(columns={"datetime": "date"})
        sub["date"] = pd.to_datetime(sub["date"]).dt.tz_localize(None)
        sub["symbol"] = sym
        sub["source"] = "yfinance"
        sub["timeframe"] = interval
        return sub[["symbol", "date", "open", "high", "low", "close", "volume", "source", "timeframe"]]

    for i in range(0, len(clean_symbols), batch_size):
        batch = clean_symbols[i : i + batch_size]
        log(f"bulk_download_yfinance()::batch {i + 1}–{min(i + batch_size, len(clean_symbols))} / {len(clean_symbols)}")

        try:
            raw = yf.download(
                batch,
                start=start_dt,
                end=end_dt,
                interval=yf_interval,
                group_by="ticker",
                auto_adjust=True,
                actions=False,
                progress=False,
            )
        except Exception as e:
            log(f"bulk_download_yfinance()::batch error: {e}")
            failures.extend((s, str(e)) for s in batch)
            continue

        if raw.empty:
            continue

        batch_frames = []

        if isinstance(raw.columns, pd.MultiIndex):
            present = set(raw.columns.get_level_values(0).unique())
            for sym in batch:
                if sym not in present:
                    failures.append((sym, "not in response"))
                    continue
                sub = raw[sym]
                if sub.dropna(how="all").empty:
                    failures.append((sym, "no data"))
                    continue
                df = _normalise(sub, sym)
                batch_frames.append(df)
                log(f"bulk_download_yfinance()::OK {sym} rows={len(df)}")
        else:
            # single-ticker batch — raw has flat columns
            sym = batch[0]
            if not raw.dropna(how="all").empty:
                df = _normalise(raw, sym)
                batch_frames.append(df)
                log(f"bulk_download_yfinance()::OK {sym} rows={len(df)}")
            else:
                failures.append((sym, "no data"))

        all_frames.extend(batch_frames)

        if save and batch_frames:
            save_ohlcv_outputs_fast(pd.concat(batch_frames, ignore_index=True))
            log(f"bulk_download_yfinance()::saved batch of {len(batch_frames)} symbols")

    if not all_frames:
        raise RuntimeError("No symbols downloaded successfully.")

    if failures:
        log("Failures:")
        for sym, err in failures:
            log(f" - {sym}: {err}")

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)
    return combined


# download symbols from massive/polygon
def fetch_massive_symbol(
    symbol: str,
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    session: Optional[requests.Session] = None,
) ->pd.DataFrame:
    """
    Download OHLCV data for one symbol from Polygon.io / Massive.com.

    Parameters
    ----------
    symbol : str
        Example: 'AAPL'
    interval : str
        'd' = daily, 'w' = weekly, 'm' = monthly
    start : str | None
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD'
    end : str | None
        End date in 'YYYYMMDD' or 'YYYY-MM-DD'
    session : requests.Session | None
        Optional shared session for performance

    Returns
    -------
    pd.DataFrame
        Columns:
        symbol, date, open, high, low, close, volume
    """

    def format_date(d: str) -> str:
        # Accept 'YYYY-MM-DD' or 'YYYYMMDD'
        return pd.to_datetime(d).strftime("%Y-%m-%d")

    symbol_upper = symbol.upper()

    match interval:
        case 'w':
            interval = 'week'
        case 'm':
            interval = 'month'
        case 'y':
            interval = 'year'
        case 'd' | _:
            interval = 'day'

    # Build URL dynamically
    log(f"symbol: {symbol_upper}, interval: {interval}, start: {start}, end: {end}")
    url = f"https://api.massive.com/v2/aggs/ticker/{symbol_upper}/range/1/{interval}/{format_date(start)}/{format_date(end)}?adjusted=true&sort=asc&limit=120&apiKey={config.massive_api_key}"

    sess = session or requests.Session()
    response = sess.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()

    if not data or "results" not in data or not data["results"]:
        raise ValueError(f"No data returned for symbol: {symbol}")
    
    df = (
        pd.DataFrame(data["results"])
        .rename(columns={
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "vw": "vwap",
            "t": "timestamp",
            "n": "transactions",
        })
    )
    # convert timestamp to date column
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    required_cols = {"date", "open", "high", "low", "close", "volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    df["symbol"] = symbol_upper
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Remove bad rows
    df = df.dropna(subset=["date"])

    # Sort ascending (Stooq returns newest first)
    df = df.sort_values("date").reset_index(drop=True)
    df["source"] = "massive"

    if interval == "day":
        df["timeframe"] = "d"

    # Reorder columns
    df = df[["symbol", "date", "open", "high", "low", "close", "volume", "source", "timeframe"]]

    return df

def bulk_download_massive(
    symbols: List[str],
    interval: str = "d",
    start: Optional[str] = None,
    end: Optional[str] = None,
    save: bool = False,
    sleep_min: float = 0.4,
    sleep_max: float = 1.2,
) -> pd.DataFrame:
    """
    Download many symbols from Massive with light throttling.

    Parameters
    ----------
    symbols : list[str]
        Example: ['AAPL', 'MSFT', 'GOOG']
    interval : str
        d / w / m
    start : str | None
        Start date (YYYYMMDD or YYYY-MM-DD)
    end : str | None
        End date (YYYYMMDD or YYYY-MM-DD)
    save : bool
        Save to database
    sleep_min, sleep_max : float
        Random delay between requests to reduce risk of blocking

    Returns
    -------
    pd.DataFrame
        Combined OHLCV dataframe for all symbols
    """

    all_frames = []
    batch_frames = []  # <-- for every 10 symbols
    failures = []

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/145.0.0.0 Safari/537.36"
                )
            }
        )

        for i, symbol in enumerate(symbols, start=1):
            try:
                df = fetch_massive_symbol(
                    symbol,
                    interval=interval,
                    start=start,
                    end=end,
                    session=session
                )

                all_frames.append(df)
                batch_frames.append(df)

                log(f"[{i}/{len(symbols)}] OK   {symbol}  rows={len(df)}")

                # 🔥 Save every 5 symbols
                if save and i % 5 == 0:
                    batch_df = pd.concat(batch_frames, ignore_index=True)
                    # save_ohlcv_outputs(batch_df)
                    save_ohlcv_outputs_fast(batch_df)
                    log(f"💾 Saved batch of 5 symbols (up to {symbol})")
                    batch_frames = []  # reset batch
                    # massive api limit, 5 request per min
                    time_module.sleep(65)

            except Exception as e:
                failures.append((symbol, str(e)))
                log(f"[{i}/{len(symbols)}] FAIL {symbol}  error={e}")

            time_module.sleep(random.uniform(sleep_min, sleep_max))

    # 🔥 Save remaining symbols (<10)
    if save and batch_frames:
        batch_df = pd.concat(batch_frames, ignore_index=True)
        # save_ohlcv_outputs(batch_df)
        save_ohlcv_outputs_fast(batch_df)
        log(f"💾 Saved final batch of {len(batch_frames)} symbols")

    if not all_frames:
        raise RuntimeError("No symbols downloaded successfully.")

    combined = pd.concat(all_frames, ignore_index=True)

    # Remove duplicates if any
    combined = combined.drop_duplicates(subset=["symbol", "date"]).reset_index(drop=True)

    if failures:
        log("\nFailures:")
        for sym, err in failures:
            log(f" - {sym}: {err}")

    return combined

# download symbols from finnhub.io
def fetch_finnhub_symbol(
    symbol: str,
    interval: str = "D",
    start: Optional[str] = None,
    end: Optional[str] = None,
    session: Optional[requests.Session] = None,
) ->pd.DataFrame:
    """
    Download OHLCV data for one symbol from Finnhub.io

    Parameters
    ----------
    symbol : str
        Example: 'AAPL'
    interval : str
        'd' = daily, 'w' = weekly, 'm' = monthly
    start : str | None
        Start date in 'YYYYMMDD' or 'YYYY-MM-DD'
    end : str | None
        End date in 'YYYYMMDD' or 'YYYY-MM-DD'
    session : requests.Session | None
        Optional shared session for performance

    Returns
    -------
    pd.DataFrame
        Columns:
        symbol, date, open, high, low, close, volume
    """

    def format_date(d: str) -> str:
        # Accept 'YYYY-MM-DD' or 'YYYYMMDD'
        # return pd.to_datetime(d).strftime("%Y-%m-%d")
        return int(pd.to_datetime(d).timestamp())

    symbol_upper = symbol.upper()

    match interval:
        case 'w':
            interval = 'week'
        case 'm':
            interval = 'month'
        case 'y':
            interval = 'year'
        case 'd' | _:
            interval = 'D'

    # Build URL dynamically
    log(f"symbol: {symbol_upper}, interval: {interval}, start: {start}, end: {end}")
    
    # Setup client
    finnhub_client = finnhub.Client(api_key=config.finnhub_api_key)

    # Stock candles
    res = finnhub_client.stock_candles(symbol_upper, interval, format_date(start), format_date(end))

    print(res)
    
    # sess = session or requests.Session()
    # response = sess.get(url, timeout=20)
    # response.raise_for_status()

    # data = response.json()

    # if not data or "results" not in data or not data["results"]:
    #     raise ValueError(f"No data returned for symbol: {symbol}")
    
    # df = (
    #     pd.DataFrame(data["results"])
    #     .rename(columns={
    #         "o": "open",
    #         "h": "high",
    #         "l": "low",
    #         "c": "close",
    #         "v": "volume",
    #         "vw": "vwap",
    #         "t": "timestamp",
    #         "n": "transactions",
    #     })
    # )
    # # convert timestamp to date column
    # df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    # # Normalize column names
    # df.columns = [c.strip().lower() for c in df.columns]

    # required_cols = {"date", "open", "high", "low", "close", "volume"}
    # missing = required_cols - set(df.columns)
    # if missing:
    #     raise ValueError(f"{symbol}: missing columns {missing}")

    # df["symbol"] = symbol_upper
    # df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # # Remove bad rows
    # df = df.dropna(subset=["date"])

    # # Sort ascending (Stooq returns newest first)
    # df = df.sort_values("date").reset_index(drop=True)
    # df["source"] = "finnhub"

    # if interval == "D":
    #     df["timeframe"] = "d"

    # # Reorder columns
    # df = df[["symbol", "date", "open", "high", "low", "close", "volume", "source", "timeframe"]]

    # return df

def save_ohlcv_outputs(
    df_input: pd.DataFrame,
    db_url: str = db_url,
    # parquet_path: str = "",
    table_name: str = "price_bars_raw",
) -> None:
    df_copy = df_input.copy()

    required_cols = {"symbol", "date", "open", "high", "low", "close", "volume", "timeframe"}
    missing = required_cols - set(df_copy.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize for DB
    df_db = df_copy.rename(columns={"date": "bar_time"}).copy()
    df_db["bar_time"] = pd.to_datetime(df_db["bar_time"]).dt.date

    # Save parquet
    # parquet_file = Path(parquet_path)
    # parquet_file.parent.mkdir(parents=True, exist_ok=True)
    # df_copy.to_parquet(parquet_file, index=False, compression="snappy")

    # Save to DB
    engine = create_engine(db_url)

    upsert_sql = f"""
    insert into {table_name} (
        symbol, bar_time, timeframe, open, high, low, close, volume, source
    ) values (
        :symbol, :bar_time, :timeframe, :open, :high, :low, :close, :volume, :source
    )
    on conflict (symbol, bar_time, timeframe, source)
    do update set
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        volume = excluded.volume;
    """

    rows = df_db.to_dict(orient="records")

    with engine.begin() as conn:
        conn.execute(text(upsert_sql), rows)

def save_ohlcv_outputs_fast(
    df_input: pd.DataFrame,
    db_url: str = db_url,
    table_name: str = "price_bars_raw",
) -> None:
    df_copy = df_input.copy()

    required_cols = {
        "symbol", "date", "open", "high", "low", "close",
        "volume", "timeframe", "source"
    }
    missing = required_cols - set(df_copy.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Normalize for DB
    df_db = df_copy.rename(columns={"date": "bar_time"}).copy()

    # If your DB column is DATE, keep .dt.date
    # If your DB column is TIMESTAMPTZ, remove .dt.date and keep full timestamp
    df_db["bar_time"] = pd.to_datetime(df_db["bar_time"]) # .dt.date

    cols = [
        "symbol", "bar_time", "timeframe", "open", "high",
        "low", "close", "volume", "source"
    ]
    df_db = df_db[cols]

    
    # Convert SQLAlchemy-style URL into psycopg connection string
    # url = make_url(db_url)
    # conninfo = (
    #    f"host={url.host} port={url.port} dbname={url.database} "
    #    f"user={url.username} password={url.password}"
    # )
    
    create_staging_sql = f"""
    SET TIME ZONE 'America/New_York';
    CREATE TEMP TABLE tmp_price_bars_raw (
        symbol TEXT NOT NULL,
        bar_time DATE NOT NULL,
        timeframe TEXT NOT NULL,
        open NUMERIC,
        high NUMERIC,
        low NUMERIC,
        close NUMERIC,
        volume NUMERIC,
        source TEXT NOT NULL
    ) ON COMMIT DROP;
    """

    upsert_sql = f"""
    SET TIME ZONE 'America/New_York';
    INSERT INTO {table_name} (
        symbol, bar_time, timeframe, open, high, low, close, volume, source, bar_time_nyse
    )
    SELECT
        symbol, bar_time, timeframe, open, high, low, close, volume, source, bar_time
    FROM tmp_price_bars_raw
    ON CONFLICT (symbol, bar_time, timeframe, source)
    DO UPDATE SET
        open = EXCLUDED.open,
        high = EXCLUDED.high,
        low = EXCLUDED.low,
        close = EXCLUDED.close,
        volume = EXCLUDED.volume,
        ingested_at = NOW();
    """

    csv_buffer = StringIO()
    df_db.to_csv(csv_buffer, index=False, header=False)
    csv_buffer.seek(0)

    copy_sql = """
    COPY tmp_price_bars_raw (
        symbol, bar_time, timeframe, open, high, low, close, volume, source
    )
    FROM STDIN WITH (FORMAT CSV)
    """

    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(create_staging_sql)

            with cur.copy(copy_sql) as copy:
                copy.write(csv_buffer.read())

            cur.execute(upsert_sql)

        conn.commit()

def find_cointegrated_pairs(price_data,
                            # alpha error
                            sig = 0.05):
    n = price_data.shape[1]
    tickers = price_data.columns
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []

    for i in range(n):
        for j in range(i + 1, n):
            S1 = price_data.iloc[ : , i]
            S2 = price_data.iloc[ : , j]
            score, pvalue, _ = coint(S1, S2)
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < sig:
                pairs.append((tickers[i], tickers[j]
                             )
                            )
                
    return score_matrix, pvalue_matrix, pairs

def backtest_pair(prices_tom,
                  prices_jerry,
                  lookback = 22,
                  entry_z = 2.0, 
                  exit_z = 0.0):
# Signal
    spread = np.log(prices_tom) - np.log(prices_jerry)
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    zscore = (spread - rolling_mean) / rolling_std
# Position
    position = pd.Series(0.0, index = spread.index)
    trades = []
    current_pos = 0
    entry_date = entry_z_val = None

# Algorithmic Decision

    for i in range(lookback, len(zscore)
                  ):
        z = zscore.iloc[i]
        date = zscore.index[i]
        if np.isnan(z):
            position.iloc[i] = current_pos
            continue

        if current_pos == 0:
            if z < -entry_z:
                current_pos, entry_date, entry_z_val = 1, date, z
            elif z > entry_z:
                current_pos, entry_date, entry_z_val = -1, date, z

        # current_pos == 1
        elif current_pos == 1 and z >= exit_z:
            trades.append(
                {"entry_date": entry_date,
                 "exit_date": date,
                 "direction": "Long Spread",
                 "entry_z": entry_z_val,
                 "exit_z": z
                }
            )
            current_pos = 0

        # current_pos == -1
        elif current_pos == -1 and z >= exit_z:
            trades.append(
                {"entry_date": entry_date,
                 "exit_date": date,
                 "direction": "Short Spread",
                 "entry_z": entry_z_val,
                 "exit_z": z
                }
            )
            current_pos = 0  

        position.iloc[i] = current_pos

    spread_change = spread.diff()

# P & L
    daily_pnl = position.shift(1) * spread_change
    cumulative_pnl = daily_pnl.cumsum()

    results =\
    (
        pd
        .DataFrame(
            {"spread": spread,
             "zscore": zscore,
             "position": position,
             "daily_pnl": daily_pnl,
             "cumulative_pnl": cumulative_pnl}
        )
    )

    return results, trades

def backtest_pair_v2(
    prices_tom,
    prices_jerry,
    beta_lookback=90,
    z_lookback=22,
    vol_lookback=22,
    entry_z=2.0,
    exit_z=0.0,
    max_z=4.0,
    aum=100000,
    risk_per_trade=0.005,
    holding_days=5,
    transaction_cost_bps=0.0,
    annual_borrow_rate_bps=50.0,
):
    """
    Backtest a pairs trading strategy with:
    - rolling hedge ratio (beta)
    - rolling spread z-score
    - rolling spread volatility
    - dynamic sizing based on z-score conviction and spread risk

    Parameters
    ----------
    prices_tom : pd.Series
        Price series for asset A
    prices_jerry : pd.Series
        Price series for asset B
    beta_lookback : int
        Rolling window for hedge ratio regression
    z_lookback : int
        Rolling window for spread mean/std
    vol_lookback : int
        Rolling window for spread volatility
    entry_z : float
        Entry threshold
    exit_z : float
        Exit threshold
    max_z : float
        Z-score where conviction reaches full size
    aum : float
        Total capital
    risk_per_trade : float
        Fraction of AUM risk budget per trade
    holding_days : int
        Used in volatility scaling
    transaction_cost_bps : float
        Cost applied on notional change, in bps
    annual_borrow_rate_bps : float
        Annual stock-borrow fee on the short leg notional, in bps.
        50 bps ≈ cheap ETB stock; 200–500 bps for moderate HTB.
        Applied daily as: abs(short_leg_dollars) * rate / 252.

    Returns
    -------
    results : pd.DataFrame
    trades : list[dict]
    """

    # ----------------------------
    # 0) Clean and align data
    # ----------------------------
    prices = pd.concat(
        [prices_tom.rename("tom"), prices_jerry.rename("jerry")],
        axis=1
    ).dropna()

    prices_tom = prices["tom"]
    prices_jerry = prices["jerry"]

    log_tom = np.log(prices_tom)
    log_jerry = np.log(prices_jerry)

    index = prices.index

    # ----------------------------
    # 1) Rolling beta
    # ----------------------------
    rolling_beta = pd.Series(index=index, dtype=float)

    for i in range(beta_lookback, len(prices)):
        y = log_tom.iloc[i - beta_lookback:i]
        x = log_jerry.iloc[i - beta_lookback:i]

        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        rolling_beta.iloc[i] = model.params.iloc[1]

    # ----------------------------
    # 2) Spread using rolling beta
    # ----------------------------
    spread = log_tom - rolling_beta * log_jerry

    rolling_mean = spread.rolling(z_lookback).mean()
    rolling_std = spread.rolling(z_lookback).std()
    zscore = (spread - rolling_mean) / rolling_std

    # ----------------------------
    # 3) Spread volatility
    # ----------------------------
    spread_change = spread.diff()
    spread_vol = spread_change.rolling(vol_lookback).std()

    # ----------------------------
    # 4) Storage
    # ----------------------------
    signal = pd.Series(0.0, index=index)         # +1 long spread, -1 short spread, 0 flat
    gross_exposure = pd.Series(0.0, index=index)
    leg_tom = pd.Series(0.0, index=index)        # signed dollar position in TOM
    leg_jerry = pd.Series(0.0, index=index)      # signed dollar position in JERRY

    trades = []

    current_pos = 0
    entry_date = None
    entry_z_val = None
    entry_beta = None
    entry_gross = 0.0
    entry_price_tom = None
    entry_price_jerry = None

    # ----------------------------
    # 5) Dynamic sizing helper
    # ----------------------------
    def conviction_from_z(z, entry_z, max_z):
        abs_z = abs(z)
        if abs_z < entry_z:
            return 0.0
        if max_z <= entry_z:
            return 1.0
        return min((abs_z - entry_z) / (max_z - entry_z), 1.0)

    def compute_legs(z, beta, spread_vol_value):
        """
        Returns:
            gross, tom_dollars, jerry_dollars, signal
        signal:
            +1 = long spread
            -1 = short spread
        """
        if pd.isna(z) or pd.isna(beta) or pd.isna(spread_vol_value):
            return 0.0, 0.0, 0.0, 0

        if spread_vol_value <= 0:
            return 0.0, 0.0, 0.0, 0

        conviction = conviction_from_z(z, entry_z, max_z)
        if conviction == 0:
            return 0.0, 0.0, 0.0, 0

        gross = (
            aum
            * risk_per_trade
            * conviction
            / (spread_vol_value * np.sqrt(holding_days))
        )

        beta_abs = abs(beta)
        weight_tom = 1.0 / (1.0 + beta_abs)
        weight_jerry = beta_abs / (1.0 + beta_abs)

        tom_size = gross * weight_tom
        jerry_size = gross * weight_jerry

        # z < 0 => spread low => long spread: long TOM, short JERRY
        if z < 0:
            return gross, +tom_size, -jerry_size, +1

        # z > 0 => spread high => short spread: short TOM, long JERRY
        return gross, -tom_size, +jerry_size, -1

    # Need enough history for all rolling components
    start_i = max(beta_lookback, z_lookback, vol_lookback)

    # ----------------------------
    # 6) Trading loop
    # ----------------------------
    for i in range(start_i, len(index)):
        date = index[i]
        z = zscore.iloc[i]
        beta = rolling_beta.iloc[i]
        vol = spread_vol.iloc[i]

        if pd.isna(z) or pd.isna(beta) or pd.isna(vol):
            signal.iloc[i] = current_pos
            if i > 0:
                leg_tom.iloc[i] = leg_tom.iloc[i - 1]
                leg_jerry.iloc[i] = leg_jerry.iloc[i - 1]
                gross_exposure.iloc[i] = gross_exposure.iloc[i - 1]
            continue

        new_gross, new_leg_tom, new_leg_jerry, proposed_signal = compute_legs(z, beta, vol)

        # Flat -> enter
        if current_pos == 0:
            if z < -entry_z:
                current_pos = +1
                entry_date = date
                entry_z_val = z
                entry_beta = beta
                entry_gross = new_gross
                entry_price_tom = prices_tom.iloc[i]
                entry_price_jerry = prices_jerry.iloc[i]

                leg_tom.iloc[i] = new_leg_tom
                leg_jerry.iloc[i] = new_leg_jerry
                gross_exposure.iloc[i] = new_gross

            elif z > entry_z:
                current_pos = -1
                entry_date = date
                entry_z_val = z
                entry_beta = beta
                entry_gross = new_gross
                entry_price_tom = prices_tom.iloc[i]
                entry_price_jerry = prices_jerry.iloc[i]

                leg_tom.iloc[i] = new_leg_tom
                leg_jerry.iloc[i] = new_leg_jerry
                gross_exposure.iloc[i] = new_gross

            else:
                leg_tom.iloc[i] = 0.0
                leg_jerry.iloc[i] = 0.0
                gross_exposure.iloc[i] = 0.0

        # Long spread -> exit when |z| reverts to within exit_z of mean
        elif current_pos == +1:
            if z >= -exit_z:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "Long Spread",
                    "entry_z": entry_z_val,
                    "exit_z": z,
                    "entry_beta": entry_beta,
                    "exit_beta": beta,
                    "entry_gross": entry_gross,
                    "entry_price_tom": entry_price_tom,
                    "entry_price_jerry": entry_price_jerry,
                    "exit_price_tom": prices_tom.iloc[i],
                    "exit_price_jerry": prices_jerry.iloc[i],
                })

                current_pos = 0
                entry_date = None
                entry_z_val = None
                entry_beta = None
                entry_gross = 0.0
                entry_price_tom = None
                entry_price_jerry = None

                leg_tom.iloc[i] = 0.0
                leg_jerry.iloc[i] = 0.0
                gross_exposure.iloc[i] = 0.0
            else:
                # re-size dynamically while position is open
                leg_tom.iloc[i] = new_leg_tom
                leg_jerry.iloc[i] = new_leg_jerry
                gross_exposure.iloc[i] = new_gross

        # Short spread -> exit when z reverts downward to exit_z
        elif current_pos == -1:
            if z <= exit_z:
                trades.append({
                    "entry_date": entry_date,
                    "exit_date": date,
                    "direction": "Short Spread",
                    "entry_z": entry_z_val,
                    "exit_z": z,
                    "entry_beta": entry_beta,
                    "exit_beta": beta,
                    "entry_gross": entry_gross,
                    "entry_price_tom": entry_price_tom,
                    "entry_price_jerry": entry_price_jerry,
                    "exit_price_tom": prices_tom.iloc[i],
                    "exit_price_jerry": prices_jerry.iloc[i],
                })

                current_pos = 0
                entry_date = None
                entry_z_val = None
                entry_beta = None
                entry_gross = 0.0
                entry_price_tom = None
                entry_price_jerry = None

                leg_tom.iloc[i] = 0.0
                leg_jerry.iloc[i] = 0.0
                gross_exposure.iloc[i] = 0.0
            else:
                # re-size dynamically while position is open
                leg_tom.iloc[i] = new_leg_tom
                leg_jerry.iloc[i] = new_leg_jerry
                gross_exposure.iloc[i] = new_gross

        signal.iloc[i] = current_pos

    # ----------------------------
    # 7) P&L from actual leg returns
    # ----------------------------
    tom_ret = prices_tom.pct_change().fillna(0.0)
    jerry_ret = prices_jerry.pct_change().fillna(0.0)

    # Use previous day's holdings for today's P&L
    tom_pos_lag = leg_tom.shift(1).fillna(0.0)
    jerry_pos_lag = leg_jerry.shift(1).fillna(0.0)

    daily_pnl = tom_pos_lag * tom_ret + jerry_pos_lag * jerry_ret

    # Transaction costs on daily notional turnover
    turnover = (
        leg_tom.diff().abs().fillna(0.0)
        + leg_jerry.diff().abs().fillna(0.0)
    )
    daily_cost = turnover * (transaction_cost_bps / 10000.0)

    # Borrow cost on the short leg: accrues daily on the lagged short notional
    short_tom   = tom_pos_lag.clip(upper=0).abs()   # positive when tom is short
    short_jerry = jerry_pos_lag.clip(upper=0).abs() # positive when jerry is short
    daily_borrow_cost = (short_tom + short_jerry) * (annual_borrow_rate_bps / 10000.0 / 252)

    daily_pnl_after_cost = daily_pnl - daily_cost - daily_borrow_cost
    cumulative_pnl = daily_pnl_after_cost.cumsum()

    # Simple return metrics
    daily_return = daily_pnl_after_cost / aum
    cumulative_return = cumulative_pnl / aum

    results = pd.DataFrame({
        "price_tom": prices_tom,
        "price_jerry": prices_jerry,
        "beta": rolling_beta,
        "spread": spread,
        "spread_mean": rolling_mean,
        "spread_std": rolling_std,
        "spread_vol": spread_vol,
        "zscore": zscore,
        "signal": signal,
        "gross_exposure": gross_exposure,
        "leg_tom": leg_tom,
        "leg_jerry": leg_jerry,
        "tom_ret": tom_ret,
        "jerry_ret": jerry_ret,
        "daily_pnl": daily_pnl,
        "daily_cost": daily_cost,
        "daily_borrow_cost": daily_borrow_cost,
        "daily_pnl_after_cost": daily_pnl_after_cost,
        "cumulative_pnl": cumulative_pnl,
        "daily_return": daily_return,
        "cumulative_return": cumulative_return,
    })

    return results, trades

def _find_best_exit_z(
    prices_wide: pd.DataFrame,
    exit_z_candidates: List[float],
    train_months: int,
    test_months: int,
    transaction_cost_bps: float,
    max_holding_bars: int,
    fallback_exit_z: float = 0.0,
) -> tuple:
    """
    Sweep exit_z at portfolio level: run walk_forward_backtest once per candidate,
    pick the value that maximises total_n_trades × Sharpe across all pairs.
    Returns (best_exit_z, best_wf_result).
    """
    best_exit_z = fallback_exit_z
    best_score = -np.inf
    best_result = None

    for ez in exit_z_candidates:
        result = walk_forward_backtest(
            prices_wide,
            train_months=train_months,
            test_months=test_months,
            exit_z=ez,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_bars=max_holding_bars,
        )
        portfolio = result.get("portfolio", pd.Series(dtype=float))
        log(f"_find_best_exit_z::exit_z={ez}, portfolio={portfolio}")
        if isinstance(portfolio, pd.Series) and not portfolio.empty and portfolio.std() > 0:
            # AUM cancels in mean/std ratio, so Sharpe is scale-invariant
            sharpe = float((portfolio.mean() * 252) / (portfolio.std() * np.sqrt(252)))
        else:
            sharpe = 0.0
        n_trades = sum(f["n_trades"] for f in result.get("folds", []))
        score = n_trades * sharpe if sharpe > 0 else -np.inf

        log(f"_find_best_exit_z::result={result}")

        log(f"_find_best_exit_z::exit_z={ez} n_trades={n_trades} sharpe={sharpe:.8f} score={score:.2f}")
        all_fold_trades = [t for f in result.get("folds", []) for t in f.get("trades", [])]
        for t in all_fold_trades:
            log(
                f"  [{t['pair_tom']}/{t['pair_jerry']}] {t['direction']}"
                f" | entry {t['entry_date']} z={t['entry_z']:.3f}"
                f" tom={t.get('entry_price_tom', float('nan')):.4f} jerry={t.get('entry_price_jerry', float('nan')):.4f}"
                f" | exit {t['exit_date']} z={t['exit_z']:.3f}"
                f" tom={t.get('exit_price_tom', float('nan')):.4f} jerry={t.get('exit_price_jerry', float('nan')):.4f}"
                f" | holding={t.get('holding_days')} pnl={t.get('pnl', 0.0):.2f}"
            )

        if score > best_score or best_result is None:
            best_score = score
            best_exit_z = ez
            best_result = result

    log(f"_find_best_exit_z::selected exit_z={best_exit_z} (score={best_score:.2f})")
    return best_exit_z, best_result


def walk_forward_backtest(
    prices_wide: pd.DataFrame,
    train_months: int = 24,
    test_months: int = 1,
    coint_sig: float = 0.05,
    missing_threshold: float = 0.30,
    beta_lookback: int = 90,
    z_lookback: int = 22,
    vol_lookback: int = 22,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    max_z: float = 4.0,
    aum: float = 100000,
    risk_per_trade: float = 0.005,
    holding_days: int = 5,
    transaction_cost_bps: float = 5.0,
    max_holding_bars: int = 20,
) -> Dict[str, Any]:
    """
    Walk-forward backtest for the pairs trading system.

    Each fold:
      1. Train window: run clustering + cointegration on prices up to cutoff.
         Pair universe is built ONLY from this window — no future data leaks.
      2. Test window: run backtest_pair_v2 on the next test_months of data
         for every selected pair, then aggregate into a portfolio P&L.

    Parameters
    ----------
    prices_wide : pd.DataFrame
        Wide daily close prices, index=date, columns=symbol.
        Must be forward-filled and free of all-NaN columns.
    train_months : int
        Number of months in each training window.
    test_months : int
        Number of months in each test (out-of-sample) window.
    coint_sig : float
        p-value threshold for Engle-Granger cointegration test.
    missing_threshold : float
        Drop symbols with more than this fraction of missing bars in the
        training window.

    Returns
    -------
    dict with keys:
        "folds"     : list of per-fold result dicts
        "portfolio" : combined daily portfolio P&L DataFrame
        "metrics"   : overall Sharpe, Calmar, max drawdown, win rate
    """

    index = prices_wide.index
    fold_results = []
    all_daily_pnl: List[pd.Series] = []

    # Build monthly period boundaries
    monthly = prices_wide.resample("ME").last()
    periods = monthly.index  # month-end dates
    log(f'walk_forward_backtest::len(periods): {len(periods.tolist())}')
    log(f'walk_forward_backtest::periods: {periods.tolist()}')

    if len(periods) < train_months + test_months:
        raise ValueError(
            f"Need at least {train_months + test_months} months of data, "
            f"got {len(periods)}."
        )

    fold_start = 0
    while fold_start + train_months + test_months <= len(periods):
        log(f'walk_forward_backtest::fold_start: {fold_start}')
        train_end_period = periods[fold_start + train_months - 1]
        test_end_period  = periods[fold_start + train_months + test_months - 1]
        log(f'walk_forward_backtest::train_end_period: {train_end_period}')
        log(f'walk_forward_backtest::test_end_period: {test_end_period}')


        train_mask = index <= train_end_period
        test_mask  = (index > train_end_period) & (index <= test_end_period)

        train_prices = prices_wide.loc[train_mask]
        test_prices  = prices_wide.loc[test_mask]

        if train_prices.empty or test_prices.empty:
            fold_start += test_months
            continue

        # ── 1. Clean training universe ──────────────────────────────────────
        missing_frac = train_prices.isnull().mean()
        keep_cols = missing_frac[missing_frac <= missing_threshold].index
        train_clean = train_prices[keep_cols].ffill().dropna(axis=1)

        if train_clean.shape[1] < 2:
            fold_start += test_months
            continue

        # ── 2. Cluster on training data only ────────────────────────────────
        daily_ret = train_clean.pct_change().dropna()
        ret_stats = pd.DataFrame({
            "Returns":    daily_ret.mean() * 252,
            "Volatility": daily_ret.std()  * np.sqrt(252),
        })
        ret_scaled = pd.DataFrame(
            scaler.fit_transform(ret_stats),
            columns=ret_stats.columns,
            index=ret_stats.index,
        )

        ap = AffinityPropagation(random_state=2026, damping=0.8)
        ap.fit(ret_scaled)
        clustered = pd.Series(index=ret_scaled.index, data=ap.labels_.flatten())
        counts = clustered.value_counts()
        valid_clusters = counts[(counts > 1) & (counts < 50)].index

        # ── 3. Cointegration on training data only ───────────────────────────
        selected_pairs: List[tuple] = []
        for cid in valid_clusters:
            tickers_in = [t for t in clustered[clustered == cid].index
                          if t in train_clean.columns]
            if len(tickers_in) < 2:
                continue
            _, _, pairs = find_cointegrated_pairs(train_clean[tickers_in], sig=coint_sig)
            selected_pairs.extend(pairs)

        if not selected_pairs:
            fold_start += test_months
            continue

        # ── 4. Backtest each pair on test window ─────────────────────────────
        fold_pair_pnls: List[pd.Series] = []
        fold_trades: List[dict] = []

        for pair_tom, pair_jerry in selected_pairs:
            if pair_tom not in test_prices.columns or pair_jerry not in test_prices.columns:
                continue

            # Prefix test window with enough train history for warmup
            warmup_bars = max(beta_lookback, z_lookback, vol_lookback) + 5
            warmup = train_prices.iloc[-warmup_bars:]
            combined = pd.concat([warmup, test_prices])

            if pair_tom not in combined.columns or pair_jerry not in combined.columns:
                continue

            p_tom   = combined[pair_tom].dropna()
            p_jerry = combined[pair_jerry].dropna()

            try:
                results, trades = backtest_pair_v2(
                    p_tom,
                    p_jerry,
                    beta_lookback=beta_lookback,
                    z_lookback=z_lookback,
                    vol_lookback=vol_lookback,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    max_z=max_z,
                    aum=aum,
                    risk_per_trade=risk_per_trade,
                    holding_days=holding_days,
                    transaction_cost_bps=transaction_cost_bps,
                )
            except Exception:
                continue
            # log(f"walk_forward_backtest::fold_start={fold_start}, results={results}")

            # Trim to test window only — warmup rows are discarded
            test_pnl = results["daily_pnl_after_cost"].loc[
                results.index > train_end_period
            ]

            # Apply max holding stop: close any position held > max_holding_bars
            signal_series = results["signal"].loc[results.index > train_end_period]
            test_pnl = _apply_holding_stop(test_pnl, signal_series, max_holding_bars)

            for t in trades:
                exit_ = pd.Timestamp(t["exit_date"])
                entry = pd.Timestamp(t["entry_date"])
                # Skip trades that closed entirely in the warmup window
                if exit_ <= train_end_period:
                    continue
                pnl_start = entry if entry > train_end_period else train_end_period
                t["pnl"] = float(
                    results["daily_pnl_after_cost"].loc[
                        (results.index > pnl_start) & (results.index <= exit_)
                    ].sum()
                )
                t["size_tom"] = float(results["leg_tom"].loc[entry]) if entry in results.index else 0.0
                t["size_jerry"] = float(results["leg_jerry"].loc[entry]) if entry in results.index else 0.0
                t["holding_days"] = (exit_.date() - max(entry, train_end_period).date()).days
                t["pair_tom"] = pair_tom
                t["pair_jerry"] = pair_jerry
                t["fold_train_end"] = str(train_end_period.date())
                fold_trades.append(t)

            if not test_pnl.empty:
                fold_pair_pnls.append(test_pnl)

        if not fold_pair_pnls:
            fold_start += test_months
            continue

        # ── 5. Aggregate portfolio P&L for this fold ─────────────────────────
        fold_portfolio = (
            pd.concat(fold_pair_pnls, axis=1)
            .fillna(0.0)
            .sum(axis=1)
        )
        all_daily_pnl.append(fold_portfolio)

        fold_results.append({
            "fold":           fold_start,
            "train_end":      str(train_end_period.date()),
            "test_end":       str(test_end_period.date()),
            "n_pairs":        len(selected_pairs),
            "n_trades":       len(fold_trades),
            "cumulative_pnl": float(fold_portfolio.sum()),
            "trades":         fold_trades,
        })

        fold_start += test_months

    if not all_daily_pnl:
        return {"folds": [], "portfolio": pd.DataFrame(), "metrics": {}}

    portfolio_pnl = pd.concat(all_daily_pnl).sort_index()
    metrics = _compute_backtest_metrics(portfolio_pnl, aum)
    log(f"\n\n>>>>>walk_forward_backtest::aum={aum}, portfolio_pnl={portfolio_pnl}")

    return {
        "folds":     fold_results,
        "portfolio": portfolio_pnl,
        "metrics":   metrics,
    }


def _apply_holding_stop(
    daily_pnl: pd.Series,
    signal: pd.Series,
    max_bars: int,
) -> pd.Series:
    """Zero out P&L after a position has been held longer than max_bars."""
    pnl = daily_pnl.copy()
    hold_count = 0
    prev_pos = 0
    for dt in pnl.index:
        pos = signal.get(dt, 0)
        if pos != 0 and pos == prev_pos:
            hold_count += 1
        elif pos != 0:
            hold_count = 1
        else:
            hold_count = 0
        if hold_count > max_bars:
            pnl[dt] = 0.0
        prev_pos = pos
    return pnl


def _compute_backtest_metrics(daily_pnl: pd.Series, aum: float) -> Dict[str, Any]:
    """Sharpe, Calmar, max drawdown, win rate from a daily P&L series."""
    if daily_pnl.empty or daily_pnl.std() == 0:
        return {}

    daily_ret = daily_pnl / aum
    ann_ret   = daily_ret.mean() * 252
    ann_vol   = daily_ret.std()  * np.sqrt(252)
    sharpe    = ann_ret / ann_vol if ann_vol > 0 else 0.0

    cum = daily_pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd   = float(drawdown.min())
    calmar   = ann_ret / abs(max_dd / aum) if max_dd != 0 else 0.0

    nonzero = daily_pnl[daily_pnl != 0]
    win_rate = float((nonzero > 0).mean()) if len(nonzero) > 0 else 0.0

    return {
        "ann_return":   round(ann_ret,  4),
        "ann_vol":      round(ann_vol,  4),
        "sharpe":       round(sharpe,   4),
        "calmar":       round(calmar,   4),
        "max_drawdown": round(max_dd,   2),
        "win_rate":     round(win_rate, 4),
        "total_pnl":    round(float(daily_pnl.sum()), 2),
        "n_days":       len(daily_pnl),
    }


@router.get("/walk_forward_backtest")
async def run_walk_forward_backtest(
    train_months: int = 24,
    test_months: int = 1,
    transaction_cost_bps: float = 5.0,
    max_holding_bars: int = 20,
):
    """Run the walk-forward backtest and return per-fold + overall metrics."""
    parquet_path = "data/df_clean.parquet"
    prices_wide = pd.read_parquet(parquet_path)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: walk_forward_backtest(
            prices_wide,
            train_months=train_months,
            test_months=test_months,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_bars=max_holding_bars,
        ),
    )

    return {
        "metrics": result["metrics"],
        "n_folds": len(result["folds"]),
        "folds":   [
            {k: v for k, v in f.items() if k != "trades"}
            for f in result["folds"]
        ],
    }


def _build_paper_trading_payload(result: Dict[str, Any], report_months: int) -> Dict[str, Any]:
    folds = result["folds"]
    if not folds:
        return {"error": "No folds produced — check that df_clean.parquet covers enough history."}

    recent_folds = folds[-report_months:]
    all_trades = [t for f in recent_folds for t in f.get("trades", [])]

    pair_stats: Dict[str, Dict] = {}
    for t in all_trades:
        key = f"{t['pair_tom']} / {t['pair_jerry']}"
        s = pair_stats.setdefault(key, {"n_trades": 0, "wins": 0, "total_pnl": 0.0})
        s["n_trades"] += 1
        pnl = t.get("pnl", 0.0)
        s["total_pnl"] += pnl
        if pnl > 0:
            s["wins"] += 1

    by_pair = sorted(
        [
            {
                "pair": k,
                "n_trades": v["n_trades"],
                "win_rate": round(v["wins"] / v["n_trades"], 4) if v["n_trades"] else 0.0,
                "total_pnl": round(v["total_pnl"], 2),
            }
            for k, v in pair_stats.items()
        ],
        key=lambda x: -x["total_pnl"],
    )

    def _fmt_date(d) -> str:
        return str(d.date()) if hasattr(d, "date") else str(d)

    trade_rows = sorted(
        [
            {
                "pair": f"{t['pair_tom']} / {t['pair_jerry']}",
                "direction": t["direction"],
                "entry_date": _fmt_date(t["entry_date"]),
                "exit_date": _fmt_date(t["exit_date"]),
                "holding_days": t.get("holding_days"),
                "entry_zscore": round(float(t["entry_z"]), 4),
                "exit_zscore": round(float(t["exit_z"]), 4),
                "entry_beta": round(float(t["entry_beta"]), 4),
                "gross_size": round(float(t["entry_gross"]), 2),
                "size_tom": round(float(t.get("size_tom", 0.0)), 2),
                "size_jerry": round(float(t.get("size_jerry", 0.0)), 2),
                "pnl": round(float(t.get("pnl", 0.0)), 2),
            }
            for t in all_trades
        ],
        key=lambda x: x["entry_date"],
    )

    total_pnl = sum(t["pnl"] for t in trade_rows)
    wins = sum(1 for t in trade_rows if t["pnl"] > 0)

    return {
        "period": {
            "test_start": recent_folds[0]["train_end"],
            "test_end": recent_folds[-1]["test_end"],
            "n_folds": len(recent_folds),
        },
        "summary": {
            "total_trades": len(trade_rows),
            "winning_trades": wins,
            "win_rate": round(wins / len(trade_rows), 4) if trade_rows else 0.0,
            "total_pnl": round(total_pnl, 2),
        },
        "metrics": result["metrics"],
        "by_pair": by_pair,
        "trades": trade_rows,
    }


def _run_paper_trading_report(
    prices_wide: pd.DataFrame,
    train_months: int,
    test_months: int,
    report_months: int,
    transaction_cost_bps: float,
    max_holding_bars: int,
    output_path: str,
) -> None:
    import json
    log(f"paper_trading_report::starting backtest with exit_z sweep {_EXIT_Z_SWEEP}, output={output_path}")
    try:
        best_exit_z, result = _find_best_exit_z(
            prices_wide,
            exit_z_candidates=_EXIT_Z_SWEEP,
            train_months=train_months,
            test_months=test_months,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_bars=max_holding_bars,
        )
        payload = _build_paper_trading_payload(result, report_months)
        payload["exit_z"] = best_exit_z
    except Exception as e:
        payload = {"error": str(e)}

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log(f"paper_trading_report::saved to {output_path} (exit_z={best_exit_z})")


@router.get("/paper_trading_report")
async def paper_trading_report(
    train_months: int = 21,
    test_months: int = 1,
    report_months: int = 3,
    transaction_cost_bps: float = 5.0,
    max_holding_bars: int = 20,
):
    parquet_path = "data/df_clean.parquet"
    prices_wide = pd.read_parquet(parquet_path)

    log(prices_wide.head(5))
    log('...')
    log(prices_wide.tail(5))

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"data/paper_trading_report_{timestamp}.json"

    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        lambda: _run_paper_trading_report(
            prices_wide,
            train_months,
            test_months,
            report_months,
            transaction_cost_bps,
            max_holding_bars,
            output_path,
        ),
    )

    return {"message": "Calculation started", "output_file": output_path}


def pivot_and_clean_data(
    df_input: pd.DataFrame,
):
    df_copy = df_input.copy()
    df_wide = df_copy.pivot(
        index="date",
        columns="symbol",
        values="close"
    )
    missing_fractions =\
    (
        df_wide
        .isnull()
        .mean() # return percentage of missing values
        .sort_values(ascending = False)
    )
    list_to_drop =\
    sorted(missing_fractions[missing_fractions > 0.30]
           .index
           .tolist()
          )
    dataset_CLEANED =\
    (
        df_wide
        .drop(columns = list_to_drop)
        .ffill()
        .dropna(axis = 1)
    )
    return dataset_CLEANED

def get_price_bars_raw():
    df_read = pd.read_parquet("data/price_bars_raw.parquet")
    if df_read is not None:
        return df_read
    else:
        raise ValueError("No price bars raw data found")

# check day of the week, if Monday to friday and not holiday, return True
def is_trading_day(date: pd.Timestamp) -> bool:
    return date.weekday() < 5 and date.date() not in holidays.US()

# check if not weekend or holiday, get previous trading day data
# if missing data, download

#get previous trading day data, account for weekend and holiday
def get_previous_trading_date(ny_now: pd.Timestamp = datetime.now(ZoneInfo("America/New_York"))) -> pd.Timestamp:
    # if time is before market open (9.30AM), use yesterday date
    is_before_open = ny_now.time() < time(9, 30)
    if is_before_open:
        ny_now = ny_now - timedelta(days=1)
    while ny_now.weekday() >= 5 or ny_now.date() in holidays.US():
        ny_now -= timedelta(days=1)
    return ny_now.date()

# check last valid trading day data 
def check_last_valid_trading_data_date_nyse() -> pd.Timestamp:
    last_valid_trading_day = get_previous_trading_date(datetime.now(ZoneInfo("America/New_York")))
    log('check_last_valid_trading_data_date_nyse::last_valid_trading_day=', {last_valid_trading_day})
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(bar_time_nyse) FROM price_bars_raw WHERE bar_time_nyse <= %s", (last_valid_trading_day,))
            result = cur.fetchone()
            last_valid_trading_data = result[0]
            return pd.to_datetime(last_valid_trading_data).date()
    return None

# get list of S&P 500 stocks
def get_SP500_stocks():
    query = "SELECT * FROM vw_current_sp500_symbols order by name"
    df = pd.read_sql(query, engine)
    return df

# sync data (takes about 2 hours) 
def _run_massive_sync():
    sp500_tickers = get_SP500_stocks()["massive_symbol"]
    sp500_tickers = sp500_tickers.dropna()
    bulk_download_massive(
        sp500_tickers,
        start=check_last_valid_trading_data_date_nyse(),
        end=get_previous_trading_date(),
        save=True,
    )

def _run_stooq_sync():
    sp500_tickers = get_SP500_stocks()["stooq_symbol"].dropna()
    sp500_tickers = sp500_tickers.dropna()
    # targets = ["brk-b.us", "bf-b.us", "cboe.us", "sndk.us"]
    # filtered = sp500_tickers[sp500_tickers.isin(targets)]
    # log(f"_run_stooq_sync()::sp500_tickers: {filtered}")
    # bulk_download_stooq(
    #     filtered,
    #     start='20180101', #check_last_valid_trading_data_date_nyse(),
    #     end=get_previous_trading_date(),
    #     save=True,
    # )

    bulk_download_stooq(
        sp500_tickers,
        start=check_last_valid_trading_data_date_nyse(),
        end=get_previous_trading_date(),
        save=True,
    )

def _run_yfinance_sync():
    sp500_tickers = get_SP500_stocks()["yfinance_symbol"].dropna().tolist()
    bulk_download_yfinance(
        sp500_tickers,
        start=check_last_valid_trading_data_date_nyse(),
        end=get_previous_trading_date(),
        save=True,
    )

@router.get("/sync_recent_data")
async def sync_recent_data():
    loop = asyncio.get_running_loop()
    # loop.run_in_executor(None, _run_massive_sync) #massive
    # loop.run_in_executor(None, _run_stooq_sync) # stooq
    loop.run_in_executor(None, _run_yfinance_sync)
    return {"message": "Sync started"}

# read trading data within the specified date range
def read_trade_data(
    start: date | None = None,
    end: date | None = None
):
    if start is None or end is None:
        last_date = check_last_valid_trading_data_date_nyse()
        last_date = pd.to_datetime(last_date)

        if start is None:
            start = (last_date - pd.DateOffset(years=2)).date()
        if end is None:
            end = last_date.date()

    isRead = False
    parquet_path = f"data/df_raw_{start}_{end}.parquet"
    try:
        df = pd.read_parquet(parquet_path)
        log(f"read_trade_data::reading trade data from '{parquet_path}'")
        isRead = True
    except Exception as e:
        query = text("""
            SELECT *
            FROM price_bars_processed
            WHERE bar_time_nyse BETWEEN :start AND :end
            
        """)
#ORDER BY symbol, bar_time_nyse
        # df = await asyncio.to_thread(
        #     lambda: pd.read_sql(
        #         query,
        #         engine,
        #         params={"start": start, "end": end}
        #     ))
        log(f"read_trade_data::reading trade data from DB")
        df = pd.read_sql(
                query,
                engine,
                params={"start": start, "end": end}
            )
    if not isRead:
        df.to_parquet(parquet_path)
        log(f"read_trade_data::writing trade data to '{parquet_path}'")
    return df

@router.get("/clean_data")
def clean_data():
    _clean_data()

def _clean_data(
    start: date | None = None,
    end: date | None = None
):
    if start is None or end is None:
        last_date = check_last_valid_trading_data_date_nyse()
        log(f'_clean_data::last_date={last_date}')
        last_date = pd.to_datetime(last_date)

        if start is None:
            start = (last_date - pd.DateOffset(years=2)).date()
        if end is None:
            end = last_date.date()

    parquet_path_clean = f"data/df_clean.parquet"
    parquet_path_clean_date = f"data/df_clean_{start}_{end}.parquet"

    # process data from last 7 days
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute("SELECT * FROM process_sp500_bars(p_since => now() - interval '7 days');")
            result = cur.fetchone()
            log(f"result: {result}")
            log(f"Inserted: {result['inserted']}")
            log(f"Updated:  {result['updated']}")
            log(f"Skipped:  {result['skipped']}")

        conn.commit()

    df_raw = read_trade_data(start = start, end = end)
    df_raw =\
    (
        df_raw.rename(columns={
            "canonical_symbol": "symbol",
            "bar_time_nyse": "date"
        })
    )

    df_raw_wide = df_raw.pivot(
        index="date",
        columns="symbol",
        values="close"
    )
    log(f"clean_data()::Last 2 years trading data:", df_raw_wide) 
    
    missing_fractions =\
    (
        df_raw_wide
        .isnull()
        .mean() # return percentage of missing values
        .sort_values(ascending = False)
    )
    
    list_to_drop =\
        sorted(missing_fractions[missing_fractions > 0.30]
            .index
            .tolist()
            )
    log(f"clean_data()::Symbols dropped due to insufficient data: {len(list_to_drop)} {list_to_drop}") 

    # drop symbols with insufficient data
    df_clean =\
    (
        df_raw_wide
        .drop(columns = list_to_drop)
    )

    # forward fill
    df_clean =\
    (
        df_clean
        .ffill()
        .dropna(axis = 1)
    )
    df_clean.to_parquet(parquet_path_clean)
    df_clean.to_parquet(parquet_path_clean_date)
    log(f"clean_data()::writing df_clean to '{parquet_path_clean}'")
    log(f"clean_data()::writing df_clean to '{parquet_path_clean_date}'")

    return df_clean

def _calculate_cluster_and_cointegration():
    # read local then db
    end_date = check_last_valid_trading_data_date_nyse()
    start_date = (end_date - pd.DateOffset(years=2)).date()

    # parquet_path = f"data/df_clean_{start_date}_{end_date}.parquet"
    parquet_path = f"data/df_clean.parquet"
    parquet_path_all_pairs = f"data/df_all_pairs.parquet"
    df_clean = None
    try:
        df_clean = pd.read_parquet(parquet_path)
        log(f"_calculate_cluster_and_cointegration::reading df_clean from '{parquet_path}'")    
    except Exception as e:
        df_clean = clean_data(start = start_date, end = end_date)

    # calculate daily returns
    daily_returns =\
    (
        df_clean
        .pct_change()
        .dropna()
    )

    # annual return/volatity
    returns =\
    (
        pd
        .DataFrame(
            {"Returns": daily_returns.mean() * 252,
            "Volatility": daily_returns.std() * np.sqrt(252)
            }
        )
    )

    returns_scaled =\
    (
        pd
        .DataFrame(
            scaler.fit_transform(returns),
            columns = returns.columns,
            index = returns.index
        )
    )

    # using AffinityPropagation to find the number of clusters
    ap = AffinityPropagation(random_state = 2026, 
                                damping = 0.8)

    ap.fit(returns_scaled)

    ap_labels =\
    (
        ap
        .labels_
    )

    number_of_cluster_AP =\
    len(
        np
        .unique(ap_labels)
    )
    log(f"Number of cluster in Affinity Propagation: {number_of_cluster_AP}")

    df_ap = returns_scaled.copy()

    df_ap["Cluster"] =\
    (
        ap_labels
        .astype(str)
    )

    df_ap =\
    (
        df_ap
        .reset_index()
        .rename(columns = {"index": "symbol"}
            )
    )

    clustered_series =\
    (
        pd
        .Series(index = returns_scaled.index,
                data = ap.labels_.flatten()
            )
    )
    counts = clustered_series.value_counts()
    for cluster_id, count in counts.sort_index().items():
        log(f"Cluster {cluster_id}: {count} symbols")

    valid_clusters = counts[(counts > 1) & (counts < 50)].index
    log(f"Pairs to evaluate in valid_clusters: {sum(counts[c] * (counts[c] - 1) // 2 for c in valid_clusters)}")

    all_pairs = []
    for clust_id in valid_clusters:
        tickers_in = [t for t in clustered_series[clustered_series == clust_id].index
                        if t in df_clean.columns]
        if len(tickers_in) < 2:
            continue
            
        score_m, pval_m, pairs = find_cointegrated_pairs(df_clean[tickers_in]
                                                        )
        cluster_dict[clust_id] = {"tickers": tickers_in,
                                "pairs": pairs}
        all_pairs.extend(pairs)

    log(f"Would you let me know--total cointegrated pairs found: {len(all_pairs)}")
    all_pairs_df = pd.DataFrame(
        all_pairs,
        columns=["pair_tom", "pair_jerry"]
    )

    all_pairs_df.to_parquet(parquet_path_all_pairs, index=False)
    log(f"_calculate_cluster_and_cointegration::writing all_pairs to '{parquet_path_all_pairs}'")

    # consider saving cluster and pair
    # df_symbol.to_parquet("data/symbols.parquet")

@router.get("/calculate_cluster_and_cointegration")
async def calculate_cluster_and_cointegration():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _calculate_cluster_and_cointegration)
    return {"message": "Calculating cluster and conintegration"}

@router.get("/calculate_candidates")
async def calculate_candidates():
    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _calculate_candidates)
    return {"message": "Calculating candidates"}

def filter_etb_pairs(pairs: List[tuple], htb_symbols: set) -> List[tuple]:
    """Remove pairs where either leg is in htb_symbols (Hard-to-Borrow)."""
    if not htb_symbols:
        return pairs
    filtered = [(t, j) for t, j in pairs if t not in htb_symbols and j not in htb_symbols]
    dropped = len(pairs) - len(filtered)
    if dropped:
        log(f"filter_etb_pairs::dropped {dropped} pair(s) due to HTB status")
    return filtered


def _calculate_candidates(htb_symbols: Optional[set] = None):
    candidates = []

    parquet_path_clean = f"data/df_clean.parquet"
    parquet_path_all_pairs = f"data/df_all_pairs.parquet"
    parquet_trade_date = pd.Timestamp.now(tz="America/New_York").date()
    parquet_path_candidate = f"data/df_candidate_{parquet_trade_date}.parquet"
    df_clean = pd.read_parquet(parquet_path_clean)
    all_pairs_df = pd.read_parquet(parquet_path_all_pairs)
    all_pairs = list(
        all_pairs_df[["pair_tom", "pair_jerry"]]
        .itertuples(index=False, name=None)
    )

    # INFRA-2: ETB filter — skip pairs where either leg is HTB
    all_pairs = filter_etb_pairs(all_pairs, htb_symbols or set())

    log(f"all_pairs: {all_pairs}")
    for pair_tom, pair_jerry in all_pairs:
        signal = get_latest_pair_signal(
            df_clean[pair_tom],
            df_clean[pair_jerry],
            beta_lookback=90,
            z_lookback=22,
            vol_lookback=22,
            entry_z=2.0,
            max_z=4.0,
            aum=100000,
            risk_per_trade=0.005,
            holding_days=5,
        )

        candidates.append({
            "pair_tom": pair_tom,
            "pair_jerry": pair_jerry,
            "signal": signal.signal,
            "zscore": signal.zscore,
            "gross_dollars": signal.gross_dollars,
            "abs_z": abs(signal.zscore),
        })

    candidates_df = pd.DataFrame(candidates)
    log(f"candidates_df: {candidates_df}")

    # save_trade_candidates_df(candidates_df, trade_date=parquet_trade_date)

    # strongest absolute z-score among tradable signals
    tradable = candidates_df[candidates_df["signal"] != 0].copy()
    if tradable.empty:
        log("_calculate_candidates()::No tradable candidates found")
        return tradable

    best = tradable.sort_values("abs_z", ascending=False).iloc[0]
    sorte = tradable.sort_values("abs_z", ascending=False)
    save_trade_candidates_df(sorte, trade_date=parquet_trade_date)
    log(f"Trade of the day (best): {best}")

    sorte.to_parquet(parquet_path_candidate)
    log(f"Trades of the day: {sorte}")
    log(f"Trades of the day: {len(sorte)}")
    return sorte

@router.get("/sync_clean_calculate_candidate")
async def sync_clean_calculate_candidate():
    """Test endpoint for algo trading service."""
    log("Good morning")
    _run_yfinance_sync()
    _clean_data()

    # Fetch HTB symbols from IBKR before scoring candidates
    htb_symbols: set = set()
    ib = None
    try:
        all_pairs_df = pd.read_parquet("data/df_all_pairs.parquet")
        all_symbols = list({sym for pair in all_pairs_df[["pair_tom", "pair_jerry"]].itertuples(index=False, name=None) for sym in pair})
        ib = await start_ibkr_app()
        etb_results = await _check_etb_status(ib, all_symbols)
        htb_symbols = {r["symbol"] for r in etb_results if r["status"] == "HTB"}
        log(f"sync_clean_calculate_candidate::HTB symbols ({len(htb_symbols)}): {sorted(htb_symbols)}")
    except Exception as e:
        log(f"sync_clean_calculate_candidate::IBKR ETB check failed: {e} — proceeding without ETB filter")
    finally:
        if ib is not None:
            stop_ibkr_app(ib)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _calculate_candidates(htb_symbols=htb_symbols))

    # print day and date
    log("Day:", datetime.now().day)
    log("Date:", datetime.now().date())
    log("Weekday:", datetime.now().weekday())
        
    # print if trading day
    log("Trading day:", is_trading_day(datetime.now()))
    
    # print if previous trading day, 
    # if today if Monday, last trading day
    log("Previous trading day:", get_previous_trading_date())
    
    # print if last valid trading day
    log(f"Last trading data:", check_last_valid_trading_data_date_nyse()) 

    # loop = asyncio.get_running_loop()
    # loop.create_task(_calculate_cluster_and_cointegration())
    return {"message": "Calculation started"}
    # print if missing data
    # return "Hello, World!"


@router.get("/calculate_candidates_etb")
async def calculate_candidates_etb():
    """
    INFRA-2: ETB-filtered candidate calculation.
    Connects to IBKR to identify HTB symbols, then builds the daily candidate
    list with those pairs removed.
    """
    # Collect all unique symbols from the pair universe
    try:
        all_pairs_df = pd.read_parquet("data/df_all_pairs.parquet")
        all_pairs = list(all_pairs_df[["pair_tom", "pair_jerry"]].itertuples(index=False, name=None))
    except Exception:
        return {"error": "df_all_pairs.parquet not found — run /calculate_cluster_and_cointegration first"}

    all_symbols = list({sym for pair in all_pairs for sym in pair})

    ib = None
    htb_symbols: set = set()
    try:
        ib = await start_ibkr_app()
        etb_results = await _check_etb_status(ib, all_symbols)
        htb_symbols = {r["symbol"] for r in etb_results if r["status"] == "HTB"}
        log(f"calculate_candidates_etb::HTB symbols ({len(htb_symbols)}): {sorted(htb_symbols)}")
    except Exception as e:
        log(f"calculate_candidates_etb::IBKR ETB check failed: {e} — proceeding without ETB filter")
    finally:
        if ib is not None:
            stop_ibkr_app(ib)

    loop = asyncio.get_running_loop()
    candidates = await loop.run_in_executor(None, lambda: _calculate_candidates(htb_symbols=htb_symbols))

    n = 0 if candidates is None or (hasattr(candidates, 'empty') and candidates.empty) else len(candidates)
    return {"message": "ETB-filtered candidates calculated", "n_candidates": n, "htb_filtered": len(htb_symbols)}


# ===========================================================
# CYCLE-1: Daily Monitor
# ===========================================================

def _check_data_freshness() -> Dict[str, Any]:
    """Compare most recent bar in DB to the expected previous trading date."""
    expected = get_previous_trading_date()
    actual = check_last_valid_trading_data_date_nyse()

    if actual is None:
        return {
            "ok": False,
            "last_date": None,
            "expected_date": str(expected),
            "days_stale": None,
            "alert": "No data found in DB",
        }

    # Count trading days between actual and expected
    days_stale = 0
    cursor = actual
    while True:
        cursor_ts = pd.Timestamp(cursor) + timedelta(days=1)
        cursor = cursor_ts.date()
        if cursor > expected:
            break
        if is_trading_day(cursor_ts):
            days_stale += 1

    ok = days_stale == 0
    return {
        "ok": ok,
        "last_date": str(actual),
        "expected_date": str(expected),
        "days_stale": days_stale,
        "alert": f"Data is {days_stale} trading day(s) stale: last={actual}, expected={expected}" if not ok else None,
    }


def _detect_regime_shift(
    dataset_cleaned: pd.DataFrame,
    short_window: int = 22,
    long_window: int = 252,
    vol_ratio_threshold: float = 1.5,
) -> Dict[str, Any]:
    """
    Flag a regime shift when short-term realized vol is significantly above the long-run baseline.
    Uses an equal-weighted universe index as proxy.
    """
    if len(dataset_cleaned) < long_window:
        return {"ok": True, "short_vol_ann": None, "long_vol_ann": None, "vol_ratio": None, "alert": None}

    eq_returns = dataset_cleaned.pct_change().dropna().mean(axis=1)

    short_vol = float(eq_returns.iloc[-short_window:].std() * np.sqrt(252))
    long_vol = float(eq_returns.iloc[-long_window:].std() * np.sqrt(252))

    if long_vol == 0:
        return {"ok": True, "short_vol_ann": None, "long_vol_ann": None, "vol_ratio": None, "alert": None}

    vol_ratio = short_vol / long_vol
    regime_alert = vol_ratio > vol_ratio_threshold

    return {
        "ok": not regime_alert,
        "short_vol_ann": round(short_vol, 4),
        "long_vol_ann": round(long_vol, 4),
        "vol_ratio": round(vol_ratio, 3),
        "alert": (
            f"Regime shift: 22d annualized vol ({short_vol:.1%}) is {vol_ratio:.1f}x "
            f"long-run vol ({long_vol:.1%})"
        ) if regime_alert else None,
    }


async def _check_position_bounds(
    ib: IB,
    dataset_cleaned: pd.DataFrame,
    max_z: float = 4.0,
    max_holding_days: int = 20,
) -> List[Dict[str, Any]]:
    """
    For each OPEN trade in DB, recompute the current z-score and check holding-day bounds.
    """
    open_trades_sql = """
        SELECT id, pair_symbol_1, pair_symbol_2, entry_time, entry_zscore, side
        FROM trades
        WHERE status = 'OPEN'
        ORDER BY entry_time DESC
    """
    open_trades = pd.read_sql(text(open_trades_sql), engine)

    if open_trades.empty:
        return []

    today = pd.Timestamp.now(tz="America/New_York").normalize()
    results = []

    for _, trade in open_trades.iterrows():
        pair_tom = trade["pair_symbol_1"]
        pair_jerry = trade["pair_symbol_2"]
        alerts = []

        if pair_tom not in dataset_cleaned.columns or pair_jerry not in dataset_cleaned.columns:
            results.append({
                "pair": f"{pair_tom}/{pair_jerry}",
                "trade_id": int(trade["id"]),
                "ok": False,
                "alerts": [f"Symbol(s) missing from price data: {pair_tom}, {pair_jerry}"],
            })
            continue

        signal = get_latest_pair_signal(
            dataset_cleaned[pair_tom],
            dataset_cleaned[pair_jerry],
        )

        if abs(signal.zscore) > max_z:
            alerts.append(f"Z-score blown out: z={signal.zscore:.2f} (max={max_z})")

        trading_days_held = None
        if trade["entry_time"] is not None:
            entry_ts = pd.Timestamp(trade["entry_time"]).tz_localize("UTC") if pd.Timestamp(trade["entry_time"]).tzinfo is None else pd.Timestamp(trade["entry_time"])
            calendar_days = (today - entry_ts.normalize()).days
            trading_days_held = max(0, calendar_days * 5 // 7)
            if trading_days_held > max_holding_days:
                alerts.append(f"Holding too long: ~{trading_days_held} trading days (max={max_holding_days})")

        results.append({
            "pair": f"{pair_tom}/{pair_jerry}",
            "trade_id": int(trade["id"]),
            "ok": len(alerts) == 0,
            "zscore": round(signal.zscore, 3),
            "trading_days_held": trading_days_held,
            "alerts": alerts,
        })

    return results


async def _check_etb_status(
    ib: IB,
    symbols: List[str],
    htb_threshold: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Request shortable shares for each symbol via IBKR.
    Symbols with shortableShares < htb_threshold are flagged as HTB.
    """
    contracts = [us_stock_contract(sym) for sym in symbols]
    await ib.qualifyContractsAsync(*contracts)

    tickers = ib.reqTickers(*contracts)
    await asyncio.sleep(2)

    results = []
    for sym, ticker in zip(symbols, tickers):
        shortable = getattr(ticker, "shortableShares", None)
        is_htb = shortable is not None and shortable < htb_threshold
        results.append({
            "symbol": sym,
            "shortable_shares": int(shortable) if shortable is not None else None,
            "status": "HTB" if is_htb else "ETB",
            "alert": f"{sym} flagged HTB: only {shortable} shares shortable" if is_htb else None,
        })

    return results


async def run_daily_monitor(
    active_pairs: List[tuple],
    dataset_cleaned: pd.DataFrame,
    max_z: float = 4.0,
    max_holding_days: int = 20,
    check_etb: bool = True,
) -> Dict[str, Any]:
    """CYCLE-1: Run all daily health checks and return a structured report."""
    report: Dict[str, Any] = {}

    report["data_freshness"] = _check_data_freshness()
    report["regime"] = _detect_regime_shift(dataset_cleaned)

    ib = None
    try:
        ib = await start_ibkr_app()

        report["position_bounds"] = await _check_position_bounds(
            ib, dataset_cleaned, max_z=max_z, max_holding_days=max_holding_days
        )

        if check_etb and active_pairs:
            all_symbols = list({sym for pair in active_pairs for sym in pair})
            report["etb_status"] = await _check_etb_status(ib, all_symbols)
        else:
            report["etb_status"] = []

    finally:
        if ib is not None:
            stop_ibkr_app(ib)

    all_alerts = []
    if report["data_freshness"].get("alert"):
        all_alerts.append(report["data_freshness"]["alert"])
    if report["regime"].get("alert"):
        all_alerts.append(report["regime"]["alert"])
    for pb in report["position_bounds"]:
        all_alerts.extend(pb.get("alerts", []))
    for etb in report.get("etb_status", []):
        if etb.get("alert"):
            all_alerts.append(etb["alert"])

    report["summary"] = {
        "ok": len(all_alerts) == 0,
        "alert_count": len(all_alerts),
        "alerts": all_alerts,
    }

    log(f"run_daily_monitor::summary: {report['summary']}")
    return report


@router.get("/run_daily_monitor")
async def daily_monitor_endpoint(
    max_z: float = 4.0,
    max_holding_days: int = 20,
    check_etb: bool = True,
):
    """CYCLE-1: Daily health checks — data freshness, position bounds, regime shift, ETB status."""
    try:
        dataset_cleaned = pd.read_parquet("data/df_clean.parquet")
    except Exception:
        return {"error": "df_clean.parquet not found — run /clean_data first"}

    try:
        all_pairs_df = pd.read_parquet("data/df_all_pairs.parquet")
        active_pairs = list(all_pairs_df[["pair_tom", "pair_jerry"]].itertuples(index=False, name=None))
    except Exception:
        active_pairs = []

    return await run_daily_monitor(
        active_pairs=active_pairs,
        dataset_cleaned=dataset_cleaned,
        max_z=max_z,
        max_holding_days=max_holding_days,
        check_etb=check_etb,
    )


# ===========================================================
# CYCLE-2: Monthly Build
# ===========================================================

_EXIT_Z_SWEEP = [0.0, 0.5, 1.0, 1.5, 2.0]

async def run_monthly_build(
    sharpe_min: float = 0.5,
    train_months: int = 24,
    test_months: int = 1,
    transaction_cost_bps: float = 5.0,
    max_holding_bars: int = 20,
) -> Dict[str, Any]:
    """
    CYCLE-2: Monthly build pipeline.
      1. Re-clean price data on the trailing 24-month window.
      2. Rerun clustering + cointegration to get a fresh pair universe.
      3. Sweep exit_z at portfolio level: pick the single exit_z that maximises
         total_n_trades × Sharpe across all pairs in the walk-forward backtest.
      4. Sharpe gate: only mark the build as approved if Sharpe >= sharpe_min.

    Returns a dict with metrics, the chosen exit_z, and 'approved' flag for CYCLE-3.
    """
    loop = asyncio.get_running_loop()

    log("run_monthly_build::step 1 — cleaning price data")
    prices_wide = await loop.run_in_executor(None, _clean_data)

    log("run_monthly_build::step 2 — clustering + cointegration")
    await loop.run_in_executor(None, _calculate_cluster_and_cointegration)

    log(f"run_monthly_build::step 3 — sweeping exit_z {_EXIT_Z_SWEEP} at portfolio level")
    best_exit_z, wf_result = await loop.run_in_executor(
        None,
        lambda: _find_best_exit_z(
            prices_wide,
            exit_z_candidates=_EXIT_Z_SWEEP,
            train_months=train_months,
            test_months=test_months,
            transaction_cost_bps=transaction_cost_bps,
            max_holding_bars=max_holding_bars,
        ),
    )

    metrics = wf_result["metrics"] if wf_result else {}
    sharpe = metrics.get("sharpe", 0.0) or 0.0
    approved = sharpe >= sharpe_min

    log(f"run_monthly_build::exit_z={best_exit_z} Sharpe={sharpe:.3f} (min={sharpe_min}) — {'APPROVED' if approved else 'REJECTED'}")

    log("run_monthly_build::step 4 — saving model version")
    try:
        all_pairs_df = pd.read_parquet("data/df_all_pairs.parquet")
        pair_list = list(all_pairs_df[["pair_tom", "pair_jerry"]].itertuples(index=False, name=None))
    except Exception:
        pair_list = []

    hyperparams = {
        "train_months": train_months,
        "test_months": test_months,
        "transaction_cost_bps": transaction_cost_bps,
        "max_holding_bars": max_holding_bars,
        "sharpe_min": sharpe_min,
        "beta_lookback": 90,
        "z_lookback": 22,
        "vol_lookback": 22,
        "entry_z": 2.0,
        "exit_z": best_exit_z,
        "exit_z_sweep": _EXIT_Z_SWEEP,
        "max_z": 4.0,
        "risk_per_trade": 0.005,
        "holding_days": 5,
    }

    version_id = await loop.run_in_executor(
        None,
        lambda: save_model_version(
            pair_list=pair_list,
            hyperparams=hyperparams,
            backtest_metrics=metrics,
            approved=approved,
        ),
    )

    return {
        "approved": approved,
        "version_id": version_id,
        "exit_z": best_exit_z,
        "sharpe_gate": {"sharpe": sharpe, "sharpe_min": sharpe_min},
        "metrics": metrics,
        "n_folds": len(wf_result["folds"]) if wf_result else 0,
        "folds": [
            {k: v for k, v in f.items() if k != "trades"}
            for f in (wf_result["folds"] if wf_result else [])
        ],
    }


@router.get("/run_monthly_build")
async def monthly_build_endpoint(
    sharpe_min: float = 0.5,
    train_months: int = 24,
    test_months: int = 1,
    transaction_cost_bps: float = 5.0,
    max_holding_bars: int = 20,
):
    """CYCLE-2: Monthly build — recluster, recointegrate, backtest, Sharpe gate."""
    return await run_monthly_build(
        sharpe_min=sharpe_min,
        train_months=train_months,
        test_months=test_months,
        transaction_cost_bps=transaction_cost_bps,
        max_holding_bars=max_holding_bars,
    )


# ===========================================================
# CYCLE-3: Version
# ===========================================================

_CREATE_MODEL_VERSIONS_SQL = """
CREATE TABLE IF NOT EXISTS model_versions (
    id              SERIAL PRIMARY KEY,
    tag             TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pair_list       JSONB NOT NULL,
    hyperparams     JSONB NOT NULL,
    backtest_metrics JSONB NOT NULL,
    n_pairs         INTEGER NOT NULL,
    approved        BOOLEAN NOT NULL DEFAULT FALSE,
    status          TEXT NOT NULL DEFAULT 'pending',
    staged_at       TIMESTAMPTZ
);
"""

_MIGRATE_MODEL_VERSIONS_SQL = """
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS staged_at TIMESTAMPTZ;
"""


def _ensure_model_versions_table() -> None:
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(_CREATE_MODEL_VERSIONS_SQL)
            cur.execute(_MIGRATE_MODEL_VERSIONS_SQL)
        conn.commit()


def save_model_version(
    pair_list: List[tuple],
    hyperparams: Dict[str, Any],
    backtest_metrics: Dict[str, Any],
    approved: bool,
    tag: Optional[str] = None,
) -> int:
    """Persist a model artifact to DB. Returns the new version id."""
    import json

    _ensure_model_versions_table()

    if tag is None:
        tag = pd.Timestamp.now(tz="America/New_York").strftime("v%Y%m%d")

    pairs_json = json.dumps([{"pair_tom": t, "pair_jerry": j} for t, j in pair_list])
    hyperparams_json = json.dumps({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in hyperparams.items()})
    metrics_json = json.dumps({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in backtest_metrics.items()})

    insert_sql = """
        INSERT INTO model_versions (tag, pair_list, hyperparams, backtest_metrics, n_pairs, approved, status)
        VALUES (%s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s)
        RETURNING id;
    """
    status = "approved" if approved else "rejected"

    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(insert_sql, (tag, pairs_json, hyperparams_json, metrics_json, len(pair_list), approved, status))
            version_id = cur.fetchone()[0]
        conn.commit()

    log(f"save_model_version::saved version id={version_id} tag={tag} approved={approved} n_pairs={len(pair_list)}")
    return version_id


@router.get("/save_model_version")
async def save_model_version_endpoint():
    """CYCLE-3: Version the latest build artifact into DB."""
    try:
        all_pairs_df = pd.read_parquet("data/df_all_pairs.parquet")
    except Exception:
        return {"error": "df_all_pairs.parquet not found — run /run_monthly_build first"}

    pair_list = list(all_pairs_df[["pair_tom", "pair_jerry"]].itertuples(index=False, name=None))

    hyperparams = {
        "train_months": 24,
        "test_months": 1,
        "beta_lookback": 90,
        "z_lookback": 22,
        "vol_lookback": 22,
        "entry_z": 2.0,
        "exit_z": 0.0,
        "max_z": 4.0,
        "risk_per_trade": 0.005,
        "holding_days": 5,
        "transaction_cost_bps": 5.0,
        "max_holding_bars": 20,
    }

    loop = asyncio.get_running_loop()
    version_id = await loop.run_in_executor(
        None,
        lambda: save_model_version(
            pair_list=pair_list,
            hyperparams=hyperparams,
            backtest_metrics={},
            approved=False,
        ),
    )

    return {"version_id": version_id, "n_pairs": len(pair_list)}


# ===========================================================
# CYCLE-4: Deploy
# ===========================================================

def get_latest_approved_version() -> Optional[Dict[str, Any]]:
    """Return the most recent approved model version from DB, or None."""
    import json

    _ensure_model_versions_table()

    query = """
        SELECT id, tag, pair_list, hyperparams, backtest_metrics, n_pairs, created_at
        FROM model_versions
        WHERE approved = TRUE AND status = 'approved'
        ORDER BY created_at DESC
        LIMIT 1;
    """
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query)
            row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "tag": row["tag"],
        "pair_list": row["pair_list"] if isinstance(row["pair_list"], list) else json.loads(row["pair_list"]),
        "hyperparams": row["hyperparams"] if isinstance(row["hyperparams"], dict) else json.loads(row["hyperparams"]),
        "backtest_metrics": row["backtest_metrics"] if isinstance(row["backtest_metrics"], dict) else json.loads(row["backtest_metrics"]),
        "n_pairs": row["n_pairs"],
        "created_at": str(row["created_at"]),
    }


def get_latest_staging_version() -> Optional[Dict[str, Any]]:
    """Return the most recent staging model version from DB, or None."""
    import json

    _ensure_model_versions_table()

    query = """
        SELECT id, tag, pair_list, hyperparams, n_pairs, staged_at
        FROM model_versions
        WHERE status = 'staging'
        ORDER BY staged_at DESC NULLS LAST
        LIMIT 1;
    """
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query)
            row = cur.fetchone()

    if row is None:
        return None

    return {
        "id": row["id"],
        "tag": row["tag"],
        "pair_list": row["pair_list"] if isinstance(row["pair_list"], list) else json.loads(row["pair_list"]),
        "hyperparams": row["hyperparams"] if isinstance(row["hyperparams"], dict) else json.loads(row["hyperparams"]),
        "n_pairs": row["n_pairs"],
        "staged_at": row["staged_at"],
    }


def _stage_version(version_id: int) -> None:
    """Set version to staging and write its pair list to df_staging_pairs.parquet."""
    import json

    select_sql = "SELECT pair_list FROM model_versions WHERE id = %s;"
    update_sql = "UPDATE model_versions SET status = 'staging', staged_at = NOW() WHERE id = %s;"

    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql, (version_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Version {version_id} not found")
            pairs = row[0] if isinstance(row[0], list) else json.loads(row[0])
            cur.execute(update_sql, (version_id,))
        conn.commit()

    pairs_df = pd.DataFrame(pairs)
    pairs_df.to_parquet("data/df_staging_pairs.parquet", index=False)
    log(f"_stage_version::staged version_id={version_id}, wrote {len(pairs_df)} pairs to df_staging_pairs.parquet")


def _check_staging_gate(
    version_id: int,
    min_staging_days: int = 5,
    max_loss_pct: float = 0.02,
) -> Dict[str, Any]:
    """
    Return gate result for a staged version.
    Passes if:
      1. staged_at is at least min_staging_days ago
      2. Alpaca paper unrealized PnL on staged pairs > -max_loss_pct of gross notional
    """
    import json
    from datetime import timezone

    _ensure_model_versions_table()

    query = "SELECT pair_list, staged_at FROM model_versions WHERE id = %s;"
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, (version_id,))
            row = cur.fetchone()

    if row is None:
        return {"passed": False, "reason": f"Version {version_id} not found"}

    staged_at = row["staged_at"]
    if staged_at is None:
        return {"passed": False, "reason": "Version has no staged_at timestamp"}

    now = pd.Timestamp.now(tz="UTC")
    if staged_at.tzinfo is None:
        staged_at = staged_at.replace(tzinfo=timezone.utc)
    days_staged = (now - pd.Timestamp(staged_at)).days

    if days_staged < min_staging_days:
        return {
            "passed": False,
            "reason": f"Only {days_staged} staging day(s); need {min_staging_days}",
            "days_staged": days_staged,
        }

    pairs = row["pair_list"] if isinstance(row["pair_list"], list) else json.loads(row["pair_list"])
    staged_symbols = {p["pair_tom"] for p in pairs} | {p["pair_jerry"] for p in pairs}

    try:
        client = start_alpaca_client(paper=config.alpaca_paper)
        positions_df = get_alpaca_positions(client)
    except Exception as e:
        log(f"_check_staging_gate::Alpaca connect failed: {e} — skipping PnL gate")
        return {"passed": True, "reason": "Alpaca unavailable; time gate passed", "days_staged": days_staged}

    staged_pos = positions_df[positions_df["symbol"].isin(staged_symbols)]
    total_unrealized_pl = float(staged_pos["unrealized_pl"].sum()) if not staged_pos.empty else 0.0
    gross_notional = float(staged_pos["market_value"].abs().sum()) if not staged_pos.empty else 0.0
    loss_ratio = total_unrealized_pl / max(gross_notional, 1.0)

    if loss_ratio < -max_loss_pct:
        return {
            "passed": False,
            "reason": f"Alpaca PnL gate failed: loss ratio {loss_ratio:.2%} < -{max_loss_pct:.2%}",
            "days_staged": days_staged,
            "unrealized_pl": total_unrealized_pl,
            "loss_ratio": loss_ratio,
        }

    return {
        "passed": True,
        "reason": "Time gate and PnL gate passed",
        "days_staged": days_staged,
        "unrealized_pl": total_unrealized_pl,
        "loss_ratio": loss_ratio,
    }


def _deploy_version(version_id: int) -> None:
    """Mark a model version as deployed and write its pair list to disk."""
    import json

    version_sql = """
        SELECT pair_list FROM model_versions WHERE id = %s;
    """
    update_sql = """
        UPDATE model_versions SET status = 'deployed' WHERE id = %s;
    """

    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor() as cur:
            cur.execute(version_sql, (version_id,))
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"Version {version_id} not found")
            pair_list_raw = row[0]
            pairs = pair_list_raw if isinstance(pair_list_raw, list) else json.loads(pair_list_raw)

            cur.execute(update_sql, (version_id,))
        conn.commit()

    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.rename(columns={"pair_tom": "pair_tom", "pair_jerry": "pair_jerry"})
    pairs_df.to_parquet("data/df_all_pairs.parquet", index=False)

    log(f"_deploy_version::deployed version_id={version_id}, wrote {len(pairs_df)} pairs to df_all_pairs.parquet")


async def run_deploy(version_id: Optional[int] = None) -> Dict[str, Any]:
    """
    CYCLE-4a: Stage the latest approved model version on Alpaca paper.
    Call /promote_to_live after min_staging_days to deploy to IBKR live.
    """
    if version_id is None:
        latest = get_latest_approved_version()
        if latest is None:
            return {"staged": False, "reason": "No approved model version found in DB"}
        version_id = latest["id"]
        tag = latest["tag"]
        n_pairs = latest["n_pairs"]
    else:
        tag = f"v{version_id}"
        n_pairs = None

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _stage_version(version_id))

    log(f"run_deploy::staged version_id={version_id} tag={tag}")
    return {
        "staged": True,
        "version_id": version_id,
        "tag": tag,
        "n_pairs": n_pairs,
        "next_step": "Call /promote_to_live after staging period completes",
    }


async def promote_to_live(
    version_id: Optional[int] = None,
    min_staging_days: int = 5,
    max_loss_pct: float = 0.02,
    skip_gate: bool = False,
) -> Dict[str, Any]:
    """
    CYCLE-4b: Promote a staged version to IBKR live after gate check.
    Gate: min_staging_days elapsed + Alpaca paper loss < max_loss_pct.
    Pass skip_gate=True to force-promote (e.g. for testing).
    """
    if version_id is None:
        staging = get_latest_staging_version()
        if staging is None:
            return {"promoted": False, "reason": "No staging version found — run /run_deploy first"}
        version_id = staging["id"]
        tag = staging["tag"]
        n_pairs = staging["n_pairs"]
    else:
        tag = f"v{version_id}"
        n_pairs = None

    if not skip_gate:
        loop = asyncio.get_running_loop()
        gate = await loop.run_in_executor(
            None,
            lambda: _check_staging_gate(version_id, min_staging_days, max_loss_pct),
        )
        if not gate["passed"]:
            return {"promoted": False, "gate": gate, "version_id": version_id}
    else:
        gate = {"passed": True, "reason": "skip_gate=True"}

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, lambda: _deploy_version(version_id))

    log(f"promote_to_live::promoted version_id={version_id} tag={tag}")
    return {
        "promoted": True,
        "version_id": version_id,
        "tag": tag,
        "n_pairs": n_pairs,
        "gate": gate,
    }


@router.get("/run_deploy")
async def deploy_endpoint(version_id: Optional[int] = None):
    """CYCLE-4a: Stage latest approved model version on Alpaca paper."""
    return await run_deploy(version_id=version_id)


@router.get("/promote_to_live")
async def promote_to_live_endpoint(
    version_id: Optional[int] = None,
    min_staging_days: int = 5,
    max_loss_pct: float = 0.02,
    skip_gate: bool = False,
):
    """CYCLE-4b: Promote staged version to IBKR live after gate check."""
    return await promote_to_live(
        version_id=version_id,
        min_staging_days=min_staging_days,
        max_loss_pct=max_loss_pct,
        skip_gate=skip_gate,
    )


@router.get("/model_versions")
async def list_model_versions_endpoint():
    """List all model versions stored in DB."""
    _ensure_model_versions_table()
    query = """
        SELECT id, tag, n_pairs, approved, status, created_at
        FROM model_versions
        ORDER BY created_at DESC
        LIMIT 50;
    """
    with psycopg.connect(psycopg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [
        {
            "id": r["id"],
            "tag": r["tag"],
            "n_pairs": r["n_pairs"],
            "approved": r["approved"],
            "status": r["status"],
            "created_at": str(r["created_at"]),
        }
        for r in rows
    ]


fetch_finnhub_symbol
@router.get("/test")
async def test():
    # process data from last 7 days
    sp500_tickers = get_SP500_stocks()["stooq_symbol"]
    

    print("test:", len(sp500_tickers))
    print("test:", len(sp500_tickers.dropna()))
    # print("test:", sp500_tickers[383:385])
  
    # with psycopg.connect(psycopg_conninfo) as conn:
    #     with conn.cursor(row_factory=dict_row) as cur:
    #         cur.execute("SELECT * FROM process_sp500_bars(p_since => now() - interval '7 days');")
    #         result = cur.fetchone()
    #         log(f"result: {result}")
    #         log(f"Inserted: {result['inserted']}")
    #         log(f"Updated:  {result['updated']}")
    #         log(f"Skipped:  {result['skipped']}")

    #     conn.commit()

    # fetch_stooq_symbol(symbol='aapl', start='2026-04-20', end='2026-04-22')
    return {
        "test": "done"
    }


# =========================================================
# 3) Reuse your strategy logic on historical data only
# =========================================================

def conviction_from_z(z: float, entry_z: float, max_z: float) -> float:
    abs_z = abs(z)
    if abs_z < entry_z:
        return 0.0
    if max_z <= entry_z:
        return 1.0
    return min((abs_z - entry_z) / (max_z - entry_z), 1.0)


def compute_pair_legs_from_signal(
    z: float,
    beta: float,
    spread_vol_value: float,
    aum: float,
    risk_per_trade: float,
    holding_days: int,
    entry_z: float,
    max_z: float,
    max_gross_dollars: Optional[float] = None,
):
    """
    Returns:
        gross_dollars, tom_dollars, jerry_dollars, signal
    """
    if pd.isna(z) or pd.isna(beta) or pd.isna(spread_vol_value):
        return 0.0, 0.0, 0.0, 0

    if spread_vol_value <= 0:
        return 0.0, 0.0, 0.0, 0

    conviction = conviction_from_z(z, entry_z, max_z)
    if conviction == 0:
        return 0.0, 0.0, 0.0, 0

    gross = (
        aum
        * risk_per_trade
        * conviction
        / (spread_vol_value * np.sqrt(holding_days))
    )

    if max_gross_dollars is not None and max_gross_dollars > 0:
        gross = min(gross, max_gross_dollars)

    beta_abs = abs(beta)
    weight_tom = 1.0 / (1.0 + beta_abs)
    weight_jerry = beta_abs / (1.0 + beta_abs)

    tom_size = gross * weight_tom
    jerry_size = gross * weight_jerry

    # z < 0 => spread low => long TOM, short JERRY
    if z < 0:
        return gross, +tom_size, -jerry_size, +1

    # z > 0 => spread high => short TOM, long JERRY
    return gross, -tom_size, +jerry_size, -1


def get_latest_pair_signal(
    prices_tom: pd.Series,
    prices_jerry: pd.Series,
    beta_lookback: int = 90,
    z_lookback: int = 22,
    vol_lookback: int = 22,
    entry_z: float = 2.0,
    exit_z: float = 0.0,   # kept for consistency, not used in entry calc
    max_z: float = 4.0,
    aum: float = 100000,
    risk_per_trade: float = 0.005,
    holding_days: int = 5,
    max_gross_dollars: Optional[float] = None,
) -> PairSignal:
    """
    Compute the most recent signal using historical closes.
    Normally pass data only up to yesterday's close.
    """

    prices = pd.concat(
        [prices_tom.rename("tom"), prices_jerry.rename("jerry")],
        axis=1
    ).dropna()

    prices_tom = prices["tom"]
    prices_jerry = prices["jerry"]

    log_tom = np.log(prices_tom)
    log_jerry = np.log(prices_jerry)
    index = prices.index

    rolling_beta = pd.Series(index=index, dtype=float)

    for i in range(beta_lookback, len(prices)):
        y = log_tom.iloc[i - beta_lookback:i]
        x = log_jerry.iloc[i - beta_lookback:i]
        x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        rolling_beta.iloc[i] = model.params.iloc[1]

    spread = log_tom - rolling_beta * log_jerry
    rolling_mean = spread.rolling(z_lookback).mean()
    rolling_std = spread.rolling(z_lookback).std()
    zscore = (spread - rolling_mean) / rolling_std

    spread_change = spread.diff()
    spread_vol = spread_change.rolling(vol_lookback).std()

    latest_idx = index[-1]
    latest_z = float(zscore.iloc[-1])
    latest_beta = float(rolling_beta.iloc[-1])
    latest_spread_vol = float(spread_vol.iloc[-1])

    gross, tom_dollars, jerry_dollars, signal = compute_pair_legs_from_signal(
        z=latest_z,
        beta=latest_beta,
        spread_vol_value=latest_spread_vol,
        aum=aum,
        risk_per_trade=risk_per_trade,
        holding_days=holding_days,
        entry_z=entry_z,
        max_z=max_z,
        max_gross_dollars=max_gross_dollars,
    )

    return PairSignal(
        signal=signal,
        zscore=latest_z,
        beta=latest_beta,
        gross_dollars=float(gross),
        tom_dollars=float(tom_dollars),
        jerry_dollars=float(jerry_dollars),
        last_price_tom=float(prices_tom.iloc[-1]),
        last_price_jerry=float(prices_jerry.iloc[-1]),
        asof=latest_idx,
    )

@router.get("/get_account_summary")
async def get_account_summary(
): 
    tags = ['AvailableFunds', 'NetLiquidation', 'BuyingPower', 'UnrealizedPnL', 'RealizedPnL']
    ib = None
    try:
        ib = await start_ibkr_app(
            host=config.ibkr_host,
            port=config.ibkr_port,
            client_id=config.ibkr_client_id,
        )
        summary_df = await get_ibkr_account_summary(ib)
        # log(summary_df.to_string())
        summary_df = summary_df[summary_df['tag'].isin(tags)]
        log(f"summary_df: {summary_df}")        

        positions_df = await refresh_ibkr_positions(ib)
        log(f"positions_df: {positions_df}")

        open_orders_df = await refresh_ibkr_open_orders(ib)
        log(f"open_orders_df: {open_orders_df}")

        return {
            "summary": summary_df.to_json(),
            "positions": positions_df.to_json(),
            "open_orders": open_orders_df.to_json()
        }
    finally: 
        if ib is not None:
            stop_ibkr_app(ib)
    

def get_open_trade_pairs() -> set:
    """Return set of (pair_symbol_1, pair_symbol_2) for all OPEN trades in DB."""
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT pair_symbol_1, pair_symbol_2 FROM trades WHERE status = 'OPEN'")
        ).fetchall()
    return {(r[0], r[1]) for r in rows}


@router.post("/trade_best_pair_live")
async def trade_best_pair_live(
    ibkr_host: str = config.ibkr_host,
    ibkr_port: int = config.ibkr_port,
    ibkr_client_id: int = config.ibkr_client_id,
    account: Optional[str] = None,
    order_type: str = "MKT",
    limit_buffer_bps: float = 10.0,
    rebalance_when_same_direction: bool = True,
    aum: float = 100000,
):
    """
    Manages the live pairs portfolio in IBKR:
    - exits/holds/rebalances all currently open pairs
    - enters new pairs (sorted by abs z-score) up to max_concurrent_pairs
    - caps each pair's gross exposure at max_pair_pct * aum
    Both limits are read from config (max_concurrent_pairs, max_pair_pct).
    """
    parquet_path = "data/df_clean.parquet"
    parquet_path_candidate = f"data/df_candidate_{datetime.now().date()}.parquet"

    df_clean = pd.read_parquet(parquet_path)
    candidates_df = pd.read_parquet(parquet_path_candidate)

    open_pairs = get_open_trade_pairs()
    n_open = len(open_pairs)
    slots_available = config.max_concurrent_pairs - n_open
    max_gross = config.max_pair_pct * aum

    shared_params = dict(
        account=account,
        beta_lookback=90,
        z_lookback=22,
        vol_lookback=22,
        entry_z=2.0,
        exit_z=0.0,
        max_z=4.0,
        aum=aum,
        risk_per_trade=0.005,
        holding_days=5,
        order_type=order_type,
        limit_buffer_bps=limit_buffer_bps,
        rebalance_when_same_direction=rebalance_when_same_direction,
        max_gross_dollars=max_gross,
    )

    ib = None
    try:
        ib = await start_ibkr_app(host=ibkr_host, port=ibkr_port, client_id=ibkr_client_id)

        results = []

        # Manage all currently open pairs (exit / hold / rebalance)
        for pair_tom, pair_jerry in open_pairs:
            if pair_tom not in df_clean.columns or pair_jerry not in df_clean.columns:
                results.append({
                    "pair_tom": pair_tom,
                    "pair_jerry": pair_jerry,
                    "action": "SKIP",
                    "reason": "Symbol(s) missing from price data",
                })
                continue
            result = await manage_live_pair(
                ib=ib,
                dataset_cleaned=df_clean,
                pair_tom=pair_tom,
                pair_jerry=pair_jerry,
                **shared_params,
            )
            results.append({"pair_tom": pair_tom, "pair_jerry": pair_jerry, **result})

        # Enter new pairs up to available slots
        slots = slots_available
        for _, row in candidates_df.iterrows():
            if slots <= 0:
                break
            pair_tom = row["pair_tom"]
            pair_jerry = row["pair_jerry"]
            if (pair_tom, pair_jerry) in open_pairs:
                continue
            if pair_tom not in df_clean.columns or pair_jerry not in df_clean.columns:
                continue
            result = await manage_live_pair(
                ib=ib,
                dataset_cleaned=df_clean,
                pair_tom=pair_tom,
                pair_jerry=pair_jerry,
                **shared_params,
            )
            results.append({"pair_tom": pair_tom, "pair_jerry": pair_jerry, **result})
            if result.get("action") == "ENTER_PAIR":
                slots -= 1

        return {
            "message": "Live trade cycle completed",
            "n_open_before": n_open,
            "slots_available": slots_available,
            "max_concurrent_pairs": config.max_concurrent_pairs,
            "max_pair_pct": config.max_pair_pct,
            "results": results,
        }

    finally:
        if ib is not None:
            stop_ibkr_app(ib)

@router.get("/preview_best_pair_live")
async def preview_best_pair_live():
    parquet_path = "data/df_clean.parquet"
    parquet_path_all_pairs = "data/df_all_pairs.parquet"
    parquet_path_candidate = f"data/df_candidate_{datetime.now().date()}.parquet"
    #parquet_path_candidate = f"data/df_candidate_2026-04-20.parquet"

    df_clean = pd.read_parquet(parquet_path)
    all_pairs_df = pd.read_parquet(parquet_path_all_pairs)
    all_pairs = list(
        all_pairs_df[["pair_tom", "pair_jerry"]]
        .itertuples(index=False, name=None)
    )

    # best = get_best_candidate_from_df(df_clean, all_pairs)
    # test = await _calculate_candidates()    
    # print(f"preview_best_pair_live::test{test}")
    # best = test.iloc[0]
    best = pd.read_parquet(parquet_path_candidate).iloc[0] #aluToDo: assumes already sorted, which is not true

    signal = get_latest_pair_signal(
        df_clean[best["pair_tom"]],
        df_clean[best["pair_jerry"]],
        beta_lookback=90,
        z_lookback=22,
        vol_lookback=22,
        entry_z=2.0,
        max_z=4.0,
        aum=100000,
        risk_per_trade=0.005,
        holding_days=5,
    )

    trade_plan = build_pair_trade_from_signal(
        pair_tom=best["pair_tom"],
        pair_jerry=best["pair_jerry"],
        signal=signal,
    )

    return {
        "best_candidate": best.to_dict(),
        "signal": {
            "signal": signal.signal,
            "zscore": signal.zscore,
            "beta": signal.beta,
            "gross_dollars": signal.gross_dollars,
            "tom_dollars": signal.tom_dollars,
            "jerry_dollars": signal.jerry_dollars,
            "last_price_tom": signal.last_price_tom,
            "last_price_jerry": signal.last_price_jerry,
            "asof": str(signal.asof),
        },
        "trade_plan": trade_plan,
    }

# =========================================================
# 4) Convert dollar targets into share quantities
# =========================================================

def round_shares(dollar_target: float, price: float) -> int:
    if price <= 0 or pd.isna(price):
        return 0
    # floor toward zero
    qty = int(abs(dollar_target) / price)
    return qty


def build_pair_trade_from_signal(
    pair_tom: str,
    pair_jerry: str,
    signal: PairSignal,
) -> Dict[str, Any]:
    """
    Build target quantities for both legs from latest signal.
    """
    qty_tom = round_shares(signal.tom_dollars, signal.last_price_tom)
    qty_jerry = round_shares(signal.jerry_dollars, signal.last_price_jerry)

    if signal.signal == 0 or qty_tom == 0 or qty_jerry == 0:
        return {
            "action": "NO_TRADE",
            "reason": "Signal is flat or computed size too small.",
            "pair_tom": pair_tom,
            "pair_jerry": pair_jerry,
            "signal": signal,
        }

    action_tom = "BUY" if signal.tom_dollars > 0 else "SELL"
    action_jerry = "BUY" if signal.jerry_dollars > 0 else "SELL"

    return {
        "action": "ENTER_PAIR",
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "signal": signal,
        "leg_tom": {
            "symbol": pair_tom,
            "action": action_tom,
            "quantity": qty_tom,
            "reference_price": signal.last_price_tom,
            "target_dollars": signal.tom_dollars,
        },
        "leg_jerry": {
            "symbol": pair_jerry,
            "action": action_jerry,
            "quantity": qty_jerry,
            "reference_price": signal.last_price_jerry,
            "target_dollars": signal.jerry_dollars,
        },
    }


# =========================================================
# 5) IBKR order helpers
# =========================================================

# ---------------------------------------------------------------------------
# DEPRECATED: The three helpers below were defined here before the IBKRApp
# class and are superseded by the ib_async versions further below.
# They are kept for reference only.
# ---------------------------------------------------------------------------
# def us_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Contract:
#     contract = Contract()
#     contract.symbol = symbol
#     contract.secType = "STK"
#     contract.exchange = exchange
#     contract.currency = currency
#     contract.primaryExchange = "NASDAQ" if symbol in {"AAPL", "MSFT", "NVDA", "AMD", "GOOG", "META"} else ""
#     return contract


# def market_order(action: str, quantity: int, account: Optional[str] = None) -> Order:
#     order = Order()
#     order.action = action
#     order.orderType = "MKT"
#     order.totalQuantity = quantity
#     if account:
#         order.account = account
#     return order


# def limit_order(action: str, quantity: int, limit_price: float, account: Optional[str] = None) -> Order:
#     order = Order()
#     order.action = action
#     order.orderType = "LMT"
#     order.totalQuantity = quantity
#     order.lmtPrice = float(limit_price)
#     if account:
#         order.account = account
#     return order
# ---------------------------------------------------------------------------
# DEPRECATED: submit_pair_trade_ibkr — superseded by submit_pair_orders which
# works with ib_async. Kept for reference only.
# ---------------------------------------------------------------------------
# def submit_pair_trade_ibkr(
#     app: IBKRApp,
#     trade_plan: Dict[str, Any],
#     account: Optional[str] = None,
#     order_type: str = "MKT",
#     limit_buffer_bps: float = 10.0,
# ) -> Dict[str, Any]:
#     """Submit the 2-leg pair trade to IBKR."""
#     if trade_plan["action"] != "ENTER_PAIR":
#         return {"submitted": False, "reason": trade_plan.get("reason", "No trade")}
#     leg_tom = trade_plan["leg_tom"]
#     leg_jerry = trade_plan["leg_jerry"]
#     submitted_orders = []
#     for leg in [leg_tom, leg_jerry]:
#         symbol = leg["symbol"]
#         action = leg["action"]
#         qty = int(leg["quantity"])
#         ref_px = float(leg["reference_price"])
#         if qty <= 0:
#             continue
#         contract = us_stock_contract(symbol)
#         if order_type.upper() == "MKT":
#             order = market_order(action, qty, account=account)
#         elif order_type.upper() == "LMT":
#             lmt_px = ref_px * (1 + limit_buffer_bps / 10000.0) if action == "BUY" else ref_px * (1 - limit_buffer_bps / 10000.0)
#             order = limit_order(action, qty, lmt_px, account=account)
#         else:
#             raise ValueError("order_type must be 'MKT' or 'LMT'")
#         order_id = app.next_order_id
#         if order_id is None:
#             raise RuntimeError("IBKR next_order_id is not available.")
#         app.placeOrder(order_id, contract, order)
#         submitted_orders.append({"order_id": order_id, "symbol": symbol, "action": action,
#                                   "quantity": qty, "order_type": order.orderType})
#         app.next_order_id += 1
#     return {"submitted": True, "orders": submitted_orders,
#             "pair_tom": trade_plan["pair_tom"], "pair_jerry": trade_plan["pair_jerry"],
#             "signal": trade_plan["signal"]}

# =========================================================
# PAPER TRADING / ALPACA
# =========================================================

try:
    from alpaca.trading.client import TradingClient as AlpacaTradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _ALPACA_AVAILABLE = True
except ImportError:
    _ALPACA_AVAILABLE = False


def start_alpaca_client(paper: bool = True) -> "AlpacaTradingClient":
    if not _ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-py not installed — run: pip install alpaca-py")
    if not config.alpaca_api_key or not config.alpaca_secret_key:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in config")
    return AlpacaTradingClient(config.alpaca_api_key, config.alpaca_secret_key, paper=paper)


def get_alpaca_positions(client: "AlpacaTradingClient") -> pd.DataFrame:
    """Return current Alpaca positions as a DataFrame."""
    positions = client.get_all_positions()
    if not positions:
        return pd.DataFrame(columns=["symbol", "qty", "side", "avg_entry_price", "market_value", "unrealized_pl"])
    return pd.DataFrame([
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "side": p.side.value,
            "avg_entry_price": float(p.avg_entry_price),
            "market_value": float(p.market_value),
            "unrealized_pl": float(p.unrealized_pl),
        }
        for p in positions
    ])


def get_alpaca_open_orders(client: "AlpacaTradingClient") -> pd.DataFrame:
    """Return open Alpaca orders as a DataFrame."""
    orders = client.get_orders()
    if not orders:
        return pd.DataFrame(columns=["id", "symbol", "qty", "side", "type", "status"])
    return pd.DataFrame([
        {
            "id": str(o.id),
            "symbol": o.symbol,
            "qty": float(o.qty) if o.qty else None,
            "side": o.side.value,
            "type": o.type.value,
            "status": o.status.value,
        }
        for o in orders
    ])


def _alpaca_symbol_position(positions_df: pd.DataFrame, symbol: str) -> float:
    """Return net position (positive=long, negative=short) for a symbol."""
    row = positions_df[positions_df["symbol"] == symbol]
    if row.empty:
        return 0.0
    qty = float(row.iloc[0]["qty"])
    side = row.iloc[0]["side"]
    return qty if side == "long" else -qty


def submit_alpaca_pair_orders(
    client: "AlpacaTradingClient",
    pair_tom: str,
    pair_jerry: str,
    signal: PairSignal,
) -> Dict[str, Any]:
    """
    Submit an Alpaca market pair trade.
    signal.signal == +1: long tom / short jerry
    signal.signal == -1: short tom / long jerry
    """
    if signal.signal == 0:
        return {"submitted": False, "reason": "No signal"}

    tom_qty = round_shares(signal.tom_dollars, signal.last_price_tom)
    jerry_qty = round_shares(signal.jerry_dollars, signal.last_price_jerry)

    if signal.signal == 1:
        tom_side = OrderSide.BUY
        jerry_side = OrderSide.SELL
    else:
        tom_side = OrderSide.SELL
        jerry_side = OrderSide.BUY

    results = []
    for symbol, qty, side in [(pair_tom, tom_qty, tom_side), (pair_jerry, jerry_qty, jerry_side)]:
        if qty <= 0:
            results.append({"symbol": symbol, "submitted": False, "reason": "qty=0"})
            continue
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        order = client.submit_order(req)
        results.append({"symbol": symbol, "submitted": True, "order_id": str(order.id), "qty": qty, "side": side.value})

    return {"submitted": True, "action": "ENTER", "orders": results}


def submit_alpaca_pair_exit_orders(
    client: "AlpacaTradingClient",
    pair_tom: str,
    pair_jerry: str,
    positions_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Close both legs of an Alpaca pair position at market."""
    results = []
    for symbol in [pair_tom, pair_jerry]:
        net_qty = _alpaca_symbol_position(positions_df, symbol)
        if net_qty == 0:
            results.append({"symbol": symbol, "submitted": False, "reason": "no position"})
            continue
        side = OrderSide.SELL if net_qty > 0 else OrderSide.BUY
        qty = abs(net_qty)
        req = MarketOrderRequest(symbol=symbol, qty=qty, side=side, time_in_force=TimeInForce.DAY)
        order = client.submit_order(req)
        results.append({"symbol": symbol, "submitted": True, "order_id": str(order.id), "qty": qty, "side": side.value})

    return {"submitted": True, "action": "EXIT", "orders": results}


def manage_alpaca_pair(
    client: "AlpacaTradingClient",
    dataset_cleaned: pd.DataFrame,
    pair_tom: str,
    pair_jerry: str,
    beta_lookback: int = 90,
    z_lookback: int = 22,
    vol_lookback: int = 22,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    max_z: float = 4.0,
    aum: float = 100000,
    risk_per_trade: float = 0.005,
    holding_days: int = 5,
) -> Dict[str, Any]:
    """Alpaca equivalent of manage_live_pair — for paper trading / staging."""
    signal = get_latest_pair_signal(
        prices_tom=dataset_cleaned[pair_tom],
        prices_jerry=dataset_cleaned[pair_jerry],
        beta_lookback=beta_lookback,
        z_lookback=z_lookback,
        vol_lookback=vol_lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        max_z=max_z,
        aum=aum,
        risk_per_trade=risk_per_trade,
        holding_days=holding_days,
    )

    positions_df = get_alpaca_positions(client)
    open_orders_df = get_alpaca_open_orders(client)

    # Skip if open orders already exist for the pair
    pair_has_orders = not open_orders_df[open_orders_df["symbol"].isin([pair_tom, pair_jerry])].empty
    if pair_has_orders:
        return {"submitted": False, "action": "SKIP", "reason": "Open orders exist", "latest_signal": signal}

    tom_pos = _alpaca_symbol_position(positions_df, pair_tom)
    jerry_pos = _alpaca_symbol_position(positions_df, pair_jerry)
    current_signal = 0
    if tom_pos > 0 and jerry_pos < 0:
        current_signal = 1
    elif tom_pos < 0 and jerry_pos > 0:
        current_signal = -1

    if current_signal == 0:
        if signal.signal == 0:
            return {"submitted": False, "action": "SKIP", "reason": "No position, no signal", "latest_signal": signal}
        return submit_alpaca_pair_orders(client, pair_tom, pair_jerry, signal)

    if signal.signal == 0 or abs(signal.zscore) <= exit_z:
        return submit_alpaca_pair_exit_orders(client, pair_tom, pair_jerry, positions_df)

    if current_signal != signal.signal:
        exit_result = submit_alpaca_pair_exit_orders(client, pair_tom, pair_jerry, positions_df)
        return {"submitted": True, "action": "EXIT_FIRST", "exit_orders": exit_result, "latest_signal": signal}

    return {"submitted": False, "action": "HOLD", "reason": "Same direction, rebalance disabled", "latest_signal": signal}


@router.get("/alpaca_paper/positions")
async def alpaca_positions_endpoint():
    """Return current Alpaca paper trading positions."""
    loop = asyncio.get_running_loop()
    client = await loop.run_in_executor(None, lambda: start_alpaca_client(paper=config.alpaca_paper))
    positions = await loop.run_in_executor(None, lambda: get_alpaca_positions(client))
    return positions.to_dict(orient="records")


@router.get("/alpaca_paper/manage_pair")
async def alpaca_manage_pair_endpoint(pair_tom: str, pair_jerry: str):
    """Alpaca paper: evaluate and execute signal for a given pair."""
    try:
        dataset_cleaned = pd.read_parquet("data/df_clean.parquet")
    except Exception:
        return {"error": "df_clean.parquet not found"}

    loop = asyncio.get_running_loop()
    client = await loop.run_in_executor(None, lambda: start_alpaca_client(paper=config.alpaca_paper))
    result = await loop.run_in_executor(
        None,
        lambda: manage_alpaca_pair(client, dataset_cleaned, pair_tom, pair_jerry),
    )
    return result


# =========================================================
# LIVE TRADING / IBKR
# =========================================================

# ---------------------------------------------------------------------------
# DEPRECATED: IBKRApp (EWrapper + EClient threading pattern)
# Replaced by ib_async IB class which handles all callbacks and event-loop
# management internally. Kept for reference only.
# ---------------------------------------------------------------------------
# class IBKRApp(EWrapper, EClient):
#     def __init__(self):
#         EClient.__init__(self, self)
#         self.next_order_id = None
#         self.errors = []
#         self.order_status_data = []
#         self.managed_accounts = []
#         self.positions = []
#         self.positions_complete = False
#         self.open_orders = []
#         self.open_orders_complete = False
#         self.account_summary = []
#         self.account_summary_complete = False
#
#     def nextValidId(self, orderId: int):
#         self.next_order_id = orderId
#         log(f"IBKR nextValidId={orderId}")
#
#     def managedAccounts(self, accountsList: str):
#         self.managed_accounts = [a.strip() for a in accountsList.split(",") if a.strip()]
#         log(f"IBKR managed accounts: {self.managed_accounts}")
#
#     def error(self, reqId, errorCode, errorString, advancedOrderRejectJson=""):
#         self.errors.append({"reqId": reqId, "errorCode": errorCode,
#                             "errorString": errorString, "advancedOrderRejectJson": advancedOrderRejectJson})
#         log(f"IBKR ERROR | reqId={reqId} code={errorCode} msg={errorString}")
#
#     def orderStatus(self, orderId, status, filled, remaining, avgFillPrice,
#                     permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
#         self.order_status_data.append({"orderId": orderId, "status": status, "filled": filled,
#                                         "remaining": remaining, "avgFillPrice": avgFillPrice,
#                                         "lastFillPrice": lastFillPrice})
#         log(f"ORDER STATUS | orderId={orderId} status={status} filled={filled} "
#             f"remaining={remaining} avgFillPrice={avgFillPrice}")
#
#     def position(self, account, contract, position, avgCost):
#         self.positions.append({"account": account, "symbol": contract.symbol,
#                                 "secType": contract.secType, "exchange": contract.exchange,
#                                 "currency": contract.currency, "position": float(position),
#                                 "avgCost": float(avgCost), "conId": getattr(contract, "conId", None)})
#
#     def positionEnd(self):
#         self.positions_complete = True
#         log("IBKR positionEnd received")
#
#     def openOrder(self, orderId, contract, order, orderState):
#         self.open_orders.append({"orderId": orderId, "symbol": contract.symbol,
#                                   "secType": contract.secType, "action": order.action,
#                                   "orderType": order.orderType, "totalQuantity": float(order.totalQuantity),
#                                   "lmtPrice": getattr(order, "lmtPrice", None),
#                                   "status": getattr(orderState, "status", None)})
#
#     def openOrderEnd(self):
#         self.open_orders_complete = True
#         log("IBKR openOrderEnd received")
#
#     def accountSummary(self, reqId, account, tag, value, currency):
#         self.account_summary.append({"account": account, "tag": tag,
#                                       "value": value, "currency": currency})
#
#     def accountSummaryEnd(self, reqId):   # NOTE: was incorrectly at module scope in original
#         self.account_summary_complete = True


async def start_ibkr_app(
    host: str = config.ibkr_host,
    port: int = config.ibkr_port,
    client_id: int = config.ibkr_client_id,
    timeout_seconds: int = 10,
) -> IB:
    """
    Connect to IBKR via ib_async.
    7497 = TWS paper default
    7496 = TWS live default
    """
    ib = IB()
    log(f"start_ibkr_app::connecting to {host}:{port} clientId={client_id}")
    await ib.connectAsync(host, port, clientId=client_id, timeout=timeout_seconds)
    # log(f"start_ibkr_app::connected, nextOrderId={ib.client.reqId}")
    return ib


def stop_ibkr_app(ib: IB):
    try:
        if ib.isConnected():
            ib.disconnect()
    except Exception as e:
        log(f"stop_ibkr_app error: {e}")


# ---------------------------------------------------------------------------
# DEPRECATED: wait_for_condition — the threading/polling pattern used by the
# old IBKRApp. ib_async handles all waiting internally via asyncio.
# ---------------------------------------------------------------------------
# def wait_for_condition(predicate, timeout_seconds: int = 10, sleep_seconds: float = 0.1):
#     end_at = time_module.time() + timeout_seconds
#     while not predicate():
#         if time_module.time() > end_at:
#             raise TimeoutError("Timed out waiting for IBKR callback condition.")
#         time_module.sleep(sleep_seconds)


async def refresh_ibkr_positions(ib: IB) -> pd.DataFrame:
    """Request and return current positions as a DataFrame."""
    positions = await ib.reqPositionsAsync()
    if not positions:
        return pd.DataFrame(columns=[
            "account", "symbol", "position", "avgCost",
            "secType", "exchange", "currency", "conId",
        ])
    rows = [
        {
            "account": pos.account,
            "symbol": pos.contract.symbol,
            "secType": pos.contract.secType,
            "exchange": pos.contract.exchange,
            "currency": pos.contract.currency,
            "position": float(pos.position),
            "avgCost": float(pos.avgCost),
            "conId": pos.contract.conId,
        }
        for pos in positions
    ]
    return pd.DataFrame(rows)


async def refresh_ibkr_open_orders(ib: IB) -> pd.DataFrame:
    """Request and return open orders as a DataFrame."""
    trades = await ib.reqOpenOrdersAsync()
    if not trades:
        return pd.DataFrame(columns=[
            "orderId", "symbol", "action", "orderType",
            "totalQuantity", "lmtPrice", "status", "secType",
        ])
    rows = [
        {
            "orderId": trade.order.orderId,
            "symbol": trade.contract.symbol,
            "secType": trade.contract.secType,
            "action": trade.order.action,
            "orderType": trade.order.orderType,
            "totalQuantity": float(trade.order.totalQuantity),
            "lmtPrice": getattr(trade.order, "lmtPrice", None),
            "status": trade.orderStatus.status if trade.orderStatus else None,
        }
        for trade in trades
    ]
    return pd.DataFrame(rows)


async def get_ibkr_account_summary(ib: IB) -> pd.DataFrame:
    """
    Request and return account summary values as a DataFrame.
    Columns: account, tag, value, currency.
    """
    values = await ib.accountSummaryAsync()
    if not values:
        return pd.DataFrame(columns=["account", "tag", "value", "currency"])
    rows = [
        {
            "account": v.account,
            "tag": v.tag,
            "value": v.value,
            "currency": v.currency,
        }
        for v in values
    ]
    return pd.DataFrame(rows)


def us_stock_contract(symbol: str, exchange: str = "SMART", currency: str = "USD") -> Stock:
    """Build an ib_async Stock contract for a US equity."""
    return Stock(symbol, exchange, currency)


def market_order(action: str, quantity: int, account: Optional[str] = None) -> Order:
    """Create a market order using ib_async Order."""
    order = MarketOrder(action, quantity)
    if account:
        order.account = account
    return order


def limit_order(action: str, quantity: int, limit_price: float, account: Optional[str] = None) -> Order:
    """Create a limit order using ib_async Order."""
    order = LimitOrder(action, quantity, round(float(limit_price), 2))
    if account:
        order.account = account
    return order


def create_order(
    action: str,
    quantity: int,
    order_type: str = "MKT",
    reference_price: Optional[float] = None,
    limit_buffer_bps: float = 10.0,
    account: Optional[str] = None,
) -> Order:
    """Factory that returns a market or limit order (ib_async Order)."""
    order_type = order_type.upper()

    if order_type == "MKT":
        return market_order(action, quantity, account=account)

    if order_type == "LMT":
        if reference_price is None or reference_price <= 0:
            raise ValueError("reference_price must be provided for LMT order")
        if action.upper() == "BUY":
            limit_price = reference_price * (1 + limit_buffer_bps / 10000.0)
        else:
            limit_price = reference_price * (1 - limit_buffer_bps / 10000.0)
        return limit_order(action, quantity, limit_price, account=account)

    raise ValueError("order_type must be 'MKT' or 'LMT'")


async def place_single_order(
    ib: IB,
    symbol: str,
    action: str,
    quantity: int,
    order_type: str = "MKT",
    reference_price: Optional[float] = None,
    limit_buffer_bps: float = 10.0,
    account: Optional[str] = None,
    order_ref: Optional[str] = None,
) -> Dict[str, Any]:
    """Place one order via ib_async and return a status dict."""
    if quantity <= 0:
        return {"submitted": False, "symbol": symbol, "reason": "quantity <= 0"}

    contract = us_stock_contract(symbol)
    order = create_order(
        action=action,
        quantity=quantity,
        order_type=order_type,
        reference_price=reference_price,
        limit_buffer_bps=limit_buffer_bps,
        account=account,
    )
    if order_ref:
        order.orderRef = order_ref

    # qualify contract so IB fills in conId etc.
    await ib.qualifyContractsAsync(contract)

    trade = ib.placeOrder(contract, order)

    return {
        "submitted": True,
        "order_id": trade.order.orderId,
        "perm_id": getattr(trade.orderStatus, "permId", None),
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "order_type": order.orderType,
        "order_ref": getattr(trade.order, "orderRef", None),
    }


def round_shares(dollar_target: float, price: float) -> int:
    if price <= 0 or pd.isna(price):
        return 0
    return int(abs(dollar_target) / price)


def build_pair_trade_from_signal(
    pair_tom: str,
    pair_jerry: str,
    signal: PairSignal,
) -> Dict[str, Any]:
    qty_tom = round_shares(signal.tom_dollars, signal.last_price_tom)
    qty_jerry = round_shares(signal.jerry_dollars, signal.last_price_jerry)

    if signal.signal == 0 or qty_tom == 0 or qty_jerry == 0:
        return {
            "action": "NO_TRADE",
            "reason": "Signal is flat or computed size too small.",
            "pair_tom": pair_tom,
            "pair_jerry": pair_jerry,
            "signal": signal,
        }

    action_tom = "BUY" if signal.tom_dollars > 0 else "SELL"
    action_jerry = "BUY" if signal.jerry_dollars > 0 else "SELL"

    return {
        "action": "ENTER_PAIR",
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "signal": signal,
        "leg_tom": {
            "symbol": pair_tom,
            "action": action_tom,
            "quantity": qty_tom,
            "reference_price": signal.last_price_tom,
            "target_dollars": signal.tom_dollars,
        },
        "leg_jerry": {
            "symbol": pair_jerry,
            "action": action_jerry,
            "quantity": qty_jerry,
            "reference_price": signal.last_price_jerry,
            "target_dollars": signal.jerry_dollars,
        },
    }


def get_symbol_position(positions_df: pd.DataFrame, symbol: str) -> float:
    if positions_df.empty:
        return 0.0

    df = positions_df[
        (positions_df["symbol"] == symbol) &
        (positions_df["secType"] == "STK")
    ]
    if df.empty:
        return 0.0
    return float(df["position"].sum())


def get_pair_position_state(
    positions_df: pd.DataFrame,
    pair_tom: str,
    pair_jerry: str,
) -> Dict[str, Any]:
    tom_pos = get_symbol_position(positions_df, pair_tom)
    jerry_pos = get_symbol_position(positions_df, pair_jerry)

    # +1 long spread = long tom / short jerry
    # -1 short spread = short tom / long jerry
    if tom_pos > 0 and jerry_pos < 0:
        pair_signal = +1
    elif tom_pos < 0 and jerry_pos > 0:
        pair_signal = -1
    elif tom_pos == 0 and jerry_pos == 0:
        pair_signal = 0
    else:
        pair_signal = 99  # broken / partial / mismatched state

    return {
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "tom_position": tom_pos,
        "jerry_position": jerry_pos,
        "pair_signal": pair_signal,
    }

def pair_has_open_orders(open_orders_df: pd.DataFrame, pair_tom: str, pair_jerry: str) -> bool:
    if open_orders_df.empty:
        return False

    df = open_orders_df[
        open_orders_df["symbol"].isin([pair_tom, pair_jerry])
    ].copy()

    if df.empty:
        return False

    active_statuses = {"ApiPending", "PendingSubmit", "PendingCancel", "PreSubmitted", "Submitted"}
    if "status" in df.columns:
        df = df[df["status"].isin(active_statuses)]

    return not df.empty

def compute_target_share_quantities(signal: PairSignal) -> Dict[str, int]:
    qty_tom = round_shares(signal.tom_dollars, signal.last_price_tom)
    qty_jerry = round_shares(signal.jerry_dollars, signal.last_price_jerry)
    return {
        "qty_tom": qty_tom,
        "qty_jerry": qty_jerry,
    }

async def submit_pair_orders(
    ib: IB,
    pair_tom: str,
    pair_jerry: str,
    signal: PairSignal,
    account: Optional[str] = None,
    order_type: str = "MKT",
    limit_buffer_bps: float = 10.0,
) -> Dict[str, Any]:
    trade_plan = build_pair_trade_from_signal(pair_tom, pair_jerry, signal)

    if trade_plan["action"] == "NO_TRADE":
        return {
            "submitted": False,
            "reason": trade_plan["reason"],
            "pair_tom": pair_tom,
            "pair_jerry": pair_jerry,
        }

    leg_tom = trade_plan["leg_tom"]
    leg_jerry = trade_plan["leg_jerry"]

    trade_meta = create_trade_record(
        pair_tom=pair_tom,
        pair_jerry=pair_jerry,
        signal=signal,
        trade_plan=trade_plan,
        risk_amount=abs(float(signal.gross_dollars)) * float(0.005),
        notes=f"Submitted via ib_async order_type={order_type}",
    )
    trade_id = trade_meta["trade_id"]
    order_ref = trade_meta["order_ref"]

    results = []
    for leg in [leg_tom, leg_jerry]:
        res = await place_single_order(
            ib=ib,
            symbol=leg["symbol"],
            action=leg["action"],
            quantity=int(leg["quantity"]),
            order_type=order_type,
            reference_price=float(leg["reference_price"]),
            limit_buffer_bps=limit_buffer_bps,
            account=account,
            order_ref=order_ref,
        )
        results.append(res)

    update_trade_after_order_submission(trade_id, results)
    mark_candidate_promoted(
        trade_date=signal.asof,
        pair_tom=pair_tom,
        pair_jerry=pair_jerry,
        trade_id=trade_id,
    )

    return {
        "submitted": True,
        "action": "ENTER_PAIR",
        "trade_id": trade_id,
        "order_ref": order_ref,
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "signal": signal,
        "orders": results,
    }


async def submit_pair_exit_orders(
    ib: IB,
    pair_tom: str,
    pair_jerry: str,
    positions_df: pd.DataFrame,
    account: Optional[str] = None,
    order_type: str = "MKT",
    tom_reference_price: Optional[float] = None,
    jerry_reference_price: Optional[float] = None,
    limit_buffer_bps: float = 10.0,
) -> Dict[str, Any]:
    tom_pos = get_symbol_position(positions_df, pair_tom)
    jerry_pos = get_symbol_position(positions_df, pair_jerry)

    orders = []

    if tom_pos != 0:
        orders.append(
            await place_single_order(
                ib=ib,
                symbol=pair_tom,
                action="SELL" if tom_pos > 0 else "BUY",
                quantity=abs(int(tom_pos)),
                order_type=order_type,
                reference_price=tom_reference_price,
                limit_buffer_bps=limit_buffer_bps,
                account=account,
            )
        )

    if jerry_pos != 0:
        orders.append(
            await place_single_order(
                ib=ib,
                symbol=pair_jerry,
                action="SELL" if jerry_pos > 0 else "BUY",
                quantity=abs(int(jerry_pos)),
                order_type=order_type,
                reference_price=jerry_reference_price,
                limit_buffer_bps=limit_buffer_bps,
                account=account,
            )
        )

    return {
        "submitted": bool(orders),
        "action": "EXIT_PAIR",
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "orders": orders,
    }


async def submit_pair_rebalance_orders(
    ib: IB,
    pair_tom: str,
    pair_jerry: str,
    signal: PairSignal,
    positions_df: pd.DataFrame,
    account: Optional[str] = None,
    order_type: str = "MKT",
    limit_buffer_bps: float = 10.0,
) -> Dict[str, Any]:
    target = compute_target_share_quantities(signal)

    target_tom_signed = target["qty_tom"] if signal.tom_dollars > 0 else -target["qty_tom"]
    target_jerry_signed = target["qty_jerry"] if signal.jerry_dollars > 0 else -target["qty_jerry"]

    current_tom = int(get_symbol_position(positions_df, pair_tom))
    current_jerry = int(get_symbol_position(positions_df, pair_jerry))

    delta_tom = target_tom_signed - current_tom
    delta_jerry = target_jerry_signed - current_jerry
    log(f"submit_pair_rebalance_orders()::delta_tom: {delta_tom}")
    log(f"submit_pair_rebalance_orders()::delta_jerry: {delta_jerry}")

    orders = []

    if delta_tom != 0:
        log(f"submit_pair_rebalance_orders()::delta_tom1: {delta_tom}")
        orders.append(
            await place_single_order(
                ib=ib,
                symbol=pair_tom,
                action="BUY" if delta_tom > 0 else "SELL",
                quantity=abs(delta_tom),
                order_type=order_type,
                reference_price=signal.last_price_tom,
                limit_buffer_bps=limit_buffer_bps,
                account=account,
            )
        )

    if delta_jerry != 0:
        log(f"submit_pair_rebalance_orders()::delta_jerry1: {delta_jerry}")
        orders.append(
            await place_single_order(
                ib=ib,
                symbol=pair_jerry,
                action="BUY" if delta_jerry > 0 else "SELL",
                quantity=abs(delta_jerry),
                order_type=order_type,
                reference_price=signal.last_price_jerry,
                limit_buffer_bps=limit_buffer_bps,
                account=account,
            )
        )

    return {
        "submitted": bool(orders),
        "action": "REBALANCE_PAIR",
        "pair_tom": pair_tom,
        "pair_jerry": pair_jerry,
        "target_tom_signed": target_tom_signed,
        "target_jerry_signed": target_jerry_signed,
        "current_tom": current_tom,
        "current_jerry": current_jerry,
        "orders": orders,
    }


async def manage_live_pair(
    ib: IB,
    dataset_cleaned: pd.DataFrame,
    pair_tom: str,
    pair_jerry: str,
    account: Optional[str] = None,
    beta_lookback: int = 90,
    z_lookback: int = 22,
    vol_lookback: int = 22,
    entry_z: float = 2.0,
    exit_z: float = 0.0,
    max_z: float = 4.0,
    aum: float = 100000,
    risk_per_trade: float = 0.005,
    holding_days: int = 5,
    order_type: str = "MKT",
    limit_buffer_bps: float = 10.0,
    rebalance_when_same_direction: bool = False,
    max_gross_dollars: Optional[float] = None,
) -> Dict[str, Any]:
    signal = get_latest_pair_signal(
        prices_tom=dataset_cleaned[pair_tom],
        prices_jerry=dataset_cleaned[pair_jerry],
        beta_lookback=beta_lookback,
        z_lookback=z_lookback,
        vol_lookback=vol_lookback,
        entry_z=entry_z,
        exit_z=exit_z,
        max_z=max_z,
        aum=aum,
        risk_per_trade=risk_per_trade,
        holding_days=holding_days,
        max_gross_dollars=max_gross_dollars,
    )

    positions_df = await refresh_ibkr_positions(ib)
    log(f"manage_live_pair()::positions_df: {positions_df}")
    open_orders_df = await refresh_ibkr_open_orders(ib)
    log(f"manage_live_pair()::open_orders_df: {open_orders_df}")

    pair_state = get_pair_position_state(positions_df, pair_tom, pair_jerry)
    log(f"manage_live_pair()::pair_state: {pair_state}")

    if pair_has_open_orders(open_orders_df, pair_tom, pair_jerry):
        return {
            "submitted": False,
            "action": "SKIP",
            "reason": "Pair already has open orders",
            "pair_state": pair_state,
            "latest_signal": signal,
        }

    current_pair_signal = pair_state["pair_signal"]
    log(f"manage_live_pair()::current_pair_signal: {current_pair_signal}")
    #return {"test":"ok"}

    # No position open
    if current_pair_signal == 0:
        if signal.signal == 0:
            return {
                "submitted": False,
                "action": "SKIP",
                "reason": "No live position and no entry signal",
                "pair_state": pair_state,
                "latest_signal": signal,
            }

        return await submit_pair_orders(
            ib=ib,
            pair_tom=pair_tom,
            pair_jerry=pair_jerry,
            signal=signal,
            account=account,
            order_type=order_type,
            limit_buffer_bps=limit_buffer_bps,
        )

    # Broken partial state
    if current_pair_signal == 99:
        return {
            "submitted": False,
            "action": "MANUAL_REVIEW",
            "reason": "Pair is in partial/mismatched state",
            "pair_state": pair_state,
            "latest_signal": signal,
        }

    # Exit condition
    if signal.signal == 0 or abs(signal.zscore) <= exit_z:
        return await submit_pair_exit_orders(
            ib=ib,
            pair_tom=pair_tom,
            pair_jerry=pair_jerry,
            positions_df=positions_df,
            account=account,
            order_type=order_type,
            tom_reference_price=signal.last_price_tom,
            jerry_reference_price=signal.last_price_jerry,
            limit_buffer_bps=limit_buffer_bps,
        )

    # Opposite direction -> close then re-enter later
    if current_pair_signal != signal.signal:
        exit_result = await submit_pair_exit_orders(
            ib=ib,
            pair_tom=pair_tom,
            pair_jerry=pair_jerry,
            positions_df=positions_df,
            account=account,
            order_type=order_type,
            tom_reference_price=signal.last_price_tom,
            jerry_reference_price=signal.last_price_jerry,
            limit_buffer_bps=limit_buffer_bps,
        )
        return {
            "submitted": True,
            "action": "EXIT_FIRST",
            "reason": "Existing pair direction opposite to latest signal",
            "pair_state": pair_state,
            "latest_signal": signal,
            "exit_orders": exit_result,
        }

    # Same direction -> optionally rebalance
    if rebalance_when_same_direction:
        return await submit_pair_rebalance_orders(
            ib=ib,
            pair_tom=pair_tom,
            pair_jerry=pair_jerry,
            signal=signal,
            positions_df=positions_df,
            account=account,
            order_type=order_type,
            limit_buffer_bps=limit_buffer_bps,
        )

    return {
        "submitted": False,
        "action": "HOLD",
        "reason": "Same direction signal and rebalance disabled",
        "pair_state": pair_state,
        "latest_signal": signal,
    }


# =================== DEBUG ===================

@router.get("/debug/parquet")
async def debug_parquet(filename: str):
    """Read and log a parquet file from the data/ directory for inspection."""
    path = os.path.join("data", filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    df = pd.read_parquet(path)

    log(f"\n=== {filename} ===")
    log(f"Shape: {df.shape}")
    log(f"Columns: {list(df.columns)}")
    log(f"dtypes:\n{df.dtypes}")
    log(f"=== data ===\n", df) 
    # print(f"Head:\n{df.head(10).to_string()}")
    # print(f"Tail:\n{df.tail(5).to_string()}")

    return {
        "filename": filename,
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "head": df.head(10).to_dict(orient="records"),
        "tail": df.tail(5).to_dict(orient="records"),
    }