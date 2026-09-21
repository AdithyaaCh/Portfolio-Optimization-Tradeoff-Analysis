from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.io import loadmat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_author_example(project_root: Path) -> pd.DataFrame:
    values = loadmat(project_root / "original_author_code" / "example_data.mat")["R0"].T
    return pd.DataFrame(values, columns=[f"Asset_{i + 1:02d}" for i in range(values.shape[1])])


def _extract_adjusted(download: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if isinstance(download.columns, pd.MultiIndex):
        if "Adj Close" in download.columns.get_level_values(0):
            adjusted = download["Adj Close"]
        elif "Close" in download.columns.get_level_values(0):
            adjusted = download["Close"]
        else:
            raise ValueError("Yahoo response has neither adjusted close nor close")
    else:
        adjusted = download[["Adj Close"] if "Adj Close" in download else ["Close"]]
        adjusted.columns = tickers[:1]
    return adjusted.reindex(columns=tickers).sort_index()


def download_or_load_nifty(config: dict, force: bool = False) -> tuple[pd.DataFrame, pd.Series, dict]:
    root = Path(config["_extension_dir"])
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    stock_path = raw_dir / "nifty50_adjusted_close.csv"
    index_path = raw_dir / "nifty50_index_adjusted_close.csv"
    manifest_path = root / "data" / "data_manifest.json"
    tickers = list(config["data"]["tickers"])
    if force or not stock_path.exists() or not index_path.exists():
        stock_download = yf.download(
            tickers,
            start=config["data"]["start"],
            end=config["data"]["end"],
            auto_adjust=False,
            actions=False,
            repair=True,
            progress=False,
            threads=True,
        )
        index_ticker = config["data"]["index_ticker"]
        index_download = yf.download(
            [index_ticker],
            start=config["data"]["start"],
            end=config["data"]["end"],
            auto_adjust=False,
            actions=False,
            repair=True,
            progress=False,
            threads=False,
        )
        prices = _extract_adjusted(stock_download, tickers)
        index_prices = _extract_adjusted(index_download, [index_ticker]).iloc[:, 0]
        completeness = prices.notna().mean()
        retained = completeness[completeness >= config["data"]["min_history_fraction"]].index.tolist()
        if len(retained) < 30:
            raise RuntimeError(f"Only {len(retained)} stocks meet the history threshold; at least 30 are required")
        prices = prices[retained].dropna(how="all").ffill(limit=3)
        prices.to_csv(stock_path, index_label="Date", float_format="%.8f")
        index_prices.rename(index_ticker).to_csv(index_path, header=True, index_label="Date", float_format="%.8f")
        manifest = {
            "source": "Yahoo Finance adjusted-close field via yfinance",
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
            "constituent_reference": "https://nsearchives.nseindia.com/content/indices/ind_nifty50list.csv",
            "selection_note": "Liquid long-history subset of current NIFTY 50 constituents; survivorship bias is unavoidable.",
            "requested_start": config["data"]["start"],
            "requested_end_exclusive": config["data"]["end"],
            "requested_tickers": tickers,
            "retained_tickers": retained,
            "completeness_before_fill": {k: float(v) for k, v in completeness.items()},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    prices = pd.read_csv(stock_path, index_col="Date", parse_dates=True)
    index_prices = pd.read_csv(index_path, index_col="Date", parse_dates=True).iloc[:, 0]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stock_file_sha256"] = sha256(stock_path)
    manifest["index_file_sha256"] = sha256(index_path)
    manifest["first_observation"] = str(prices.index.min().date())
    manifest["last_observation"] = str(prices.index.max().date())
    manifest["observations"] = int(len(prices))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return prices, index_prices, manifest


def adjusted_close_returns(prices: pd.DataFrame, index_prices: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    common = prices.index.intersection(index_prices.index)
    prices = prices.reindex(common).dropna(axis=0, how="any")
    index_prices = index_prices.reindex(prices.index).ffill()
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")
    index_returns = index_prices.pct_change(fill_method=None).reindex(returns.index)
    valid = index_returns.notna()
    return returns.loc[valid], index_returns.loc[valid]


def month_end_positions(index: pd.DatetimeIndex, minimum_position: int) -> list[int]:
    frame = pd.Series(np.arange(len(index)), index=index)
    positions = frame.groupby(index.to_period("M")).last().astype(int).tolist()
    return [p for p in positions if p >= minimum_position and p < len(index) - 1]
