"""
=============================================================
  퀀트멘털(Quantamental) v8.9.2 — 리얼타임 AI 뉴스 통합 엔진
  [업데이트] 클로드 스타일 심층 마켓 브리핑 및 클라우드(GitHub) 저장 최적화
=============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import sys, os, time, io, traceback, random, json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# ── 패키지 자동 설치 ───────────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

for _mod, _pkg in [("yfinance", "yfinance"), ("pandas", "pandas"), ("numpy", "numpy"), 
                   ("jinja2", "jinja2"), ("lxml", "lxml"), ("pyarrow", "pyarrow"), 
                   ("google.generativeai", "google-generativeai")]:
    try:
        __import__(_mod)
    except ImportError:
        print(f"  📦 {_pkg} 설치 중...")
        _pip(_pkg)

import numpy as np
import pandas as pd
import yfinance as yf
import google.generativeai as genai
from jinja2 import Template

# 🌟 깃허브 비밀 금고(Secrets)에서 API 키를 안전하게 불러옵니다.
secret_key = os.environ.get("GEMINI_API_KEY")
GEMINI_API_KEYS = [secret_key] if secret_key else []
TOTAL_BUDGET = 55
INSIDER_DAYS = 90
BATCH_SIZE = 7

# ─────────────────────────────────────────────────────────────
#  1. 나스닥 100 리스트 및 데이터 수집 
# ─────────────────────────────────────────────────────────────
def get_latest_nasdaq100():
    print("  🌐 최신 나스닥 100 편입 종목 목록 확인 중...")
    tickers = []
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/Nasdaq-100", 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        html_data = urllib.request.urlopen(req, timeout=10).read().decode('utf-8')
        tables = pd.read_html(io.StringIO(html_data))
        for table in tables:
            if 'Ticker' in table.columns:
                tickers = [str(t).replace('.', '-') for t in table['Ticker'].tolist()]
                break
    except Exception as e:
        print(f"  ⚠️ 위키피디아 크롤링 실패: {e}")
    
    if tickers:
        print(f"  ✅ 최신 종목 {len(tickers)}개 업데이트 완료!")
        return tickers
        
    print("  📂 웹 크롤링 실패. 누적된 데이터 장부에서 최근 종목을 추출합니다.")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
        parquet_path = os.path.join(script_dir, "data", "nasdaq100data.parquet")
        csv_path = os.path.join(script_dir, "data", "nasdaq100data.csv")
        
        df = None
        if os.path.exists(parquet_path):
            df = pd.read_parquet(parquet_path, columns=['date', 'ticker'])
        elif os.path.exists(csv_path):
            df = pd.read_csv(csv_path, usecols=['date', 'ticker'], encoding="utf-8-sig")
            
        if df is not None and not df.empty:
            latest_date = df['date'].max()
            backup_tickers = df[df['date'] == latest_date]['ticker'].unique().tolist()
            if len(backup_tickers) > 50:
                print(f"  ✅ AI 장부 마지막 저장일({latest_date}) 기준 {len(backup_tickers)}개 종목 복구 완료!")
                return backup_tickers
    except Exception as e:
        print(f"  ⚠️ 데이터 추출 실패: {e}")
        
    print("\n  🚨 [치명적 오류] 가용 데이터가 전혀 없습니다.\n")
    return []

def _safe_float(v, default=None):
    try:
        if v is None: return default
        f = float(v)
        return default if (f != f) else f
    except: return default

def _extract_ohlcv(raw, tkr, n_tickers):
    NEED = ["Open", "High", "Low", "Close", "Volume"]
    try:
        if n_tickers == 1:
            if isinstance(raw.columns, pd.MultiIndex):
                try: return raw.xs(tkr, axis=1, level=1)[[c for c in NEED if c in raw.xs(tkr, axis=1, level=1).columns]].copy()
                except:
                    raw = raw.copy()
                    raw.columns = raw.columns.get_level_values(0)
            if "Close" in raw.columns: return raw[[c for c in NEED if c in raw.columns]].copy()
            return None

        if not isinstance(raw.columns, pd.MultiIndex): return None
        try:
            sub = raw.xs(tkr, axis=1, level=1)
            if "Close" in sub.columns: return sub[[c for c in NEED if c in sub.columns]].copy()
        except: pass
        try:
            sub = raw[tkr]
            if isinstance(sub, pd.DataFrame) and "Close" in sub.columns: return sub[[c for c in NEED if c in sub.columns]].copy()
        except: pass
        return None
    except Exception as e: 
        return None

def fetch_price_batch(batch):
    stock_data = {}
    try:
        raw = yf.download(batch, period="1y", auto_adjust=True, group_by="ticker", threads=False, progress=False)
        if raw is not None and not raw.empty:
            for tkr in batch:
                df = _extract_ohlcv(raw, tkr, len(batch))
                if df is not None:
                    df = df.dropna(subset=["Close"])
                    if len(df) >= 50: stock_data[tkr] = df
    except Exception as e:
        pass
    
    missed = [t for t in batch if t not in stock_data]
    for tkr in missed:
        try:
            df = yf.Ticker(tkr).history(period="1y", auto_adjust=True)
            if df is None or df.empty: continue
            cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
            df = df[cols].dropna(subset=["Close"])
            if len(df) >= 50: stock_data[tkr] = df
        except: pass
        time.sleep(0.15)
    return stock_data

def fetch_fundamentals(ticker_list):
    fundamentals = {}
    for tkr in ticker_list:
        row = {"short_name": tkr}
        try:
            t = yf.Ticker(tkr)
            fi = t.fast_info if hasattr(t, "fast_info") else None
            try:
                row.update({
                    "current_price": _safe_float(fi.last_price) if fi else None, 
                    "market_cap": _safe_float(fi.market_cap) if fi else None,
                    "year_high": _safe_float(fi.year_high) if fi else None
                })
            except: pass
            try:
                info = t.info or {}
                row.update({
                    "short_name":      info.get("shortName", tkr),
                    "sector":          info.get("sector", ""),
                    "earnings_growth": _safe_float(info.get("earningsGrowth")),
                    "revenue_growth":  _safe_float(info.get("revenueGrowth")),
                    "profit_margin":   _safe_float(info.get("profitMargins")),
                    "analyst_target":  _safe_float(info.get("targetMeanPrice")),
                    "peg_ratio":       _safe_float(info.get("trailingPegRatio") or info.get("pegRatio")),
                    "52w_high":        _safe_float(info.get("fiftyTwoWeekHigh")) or row.get("year_high"),
                    "fcf":             _safe_float(info.get("freeCashflow")),            
                    "roe":             _safe_float(info.get("returnOnEquity")),           
                    "inst_pct":        _safe_float(info.get("heldPercentInstitutions")), 
                    "short_pct":       _safe_float(info.get("shortPercentOfFloat")),     
                })
                if not row.get("market_cap"): row["market_cap"] = _safe_float(info.get("marketCap"))
            except: pass
        except: pass
        fundamentals[tkr] = row
        time.sleep(0.2)
    return fundamentals

def fetch_market_env():
    env = {}
    for sym in ["QQQ", "^VIX", "SPY", "^IXIC", "^TNX"]:
        try:
            df = yf.Ticker(sym).history(period="1y", auto_adjust=True)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
                if "Close" in df.columns: env[sym] = df
        except: pass
    return env

def fetch_insider_and_gov(ticker_list):
    insider_data = {}
    for tkr in ticker_list:
        try:
            df = yf.Ticker(tkr).insider_transactions
            insider_data[tkr] = df if (df is not None and not df.empty) else None
        except: insider_data[tkr] = None
        time.sleep(0.12)
    return insider_data, {}

# ─────────────────────────────────────────────────────────────
#  2. 기술 지표 계산
# ─────────────────────────────────────────────────────────────
def _to_series(df, col):
    s = df[col]
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    return s.astype(float).dropna()

def _rsi(s, period=14):
    d = s.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def _macd_diff(s, fast=12, slow=26, signal=9):
    macd = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    sig = macd.ewm(span=signal, adjust=False).mean()
    val = float((macd - sig).iloc[-1])
    return val if val == val else 0.0

def _bb_bands(s, period=20, n_std=2):
    mid = s.rolling(period).mean()
    sigma = s.rolling(period).std()
    upper = float((mid + n_std * sigma).iloc[-1])
    lower = float((mid - n_std * sigma).iloc[-1])
    return (upper if upper==upper else None, lower if lower==lower else None)

def calc_technicals(df):
    try:
        close, vol = _to_series(df, "Close"), _to_series(df, "Volume")
        if len(close) < 50: return None
        tech, price = {}, float(close.iloc[-1])
        tech["price"] = price

        for n in [20, 60, 120, 200]:
            tech[f"ma{n}"] = float(close.rolling(n).mean().iloc[-1]) if len(close) >= n else None

        tech["rsi"] = float(_rsi(close).iloc[-1])
        tech["macd_diff"] = _macd_diff(close) if len(close) >= 35 else 0.0
        if len(close) >= 20: tech["bb_upper"], tech["bb_lower"] = _bb_bands(close)
        else: tech["bb_upper"] = tech["bb_lower"] = None

        vc = vol.replace(0, np.nan).dropna()
        if len(vc) >= 20:
            avg, cur = float(vc.rolling(20).mean().iloc[-1]), float(vc.iloc[-1])
            tech["vol_ratio"] = round(cur / avg, 2) if avg > 0 else 1.0
        else: tech["vol_ratio"] = 1.0

        prev = float(close.iloc[-2]) if len(close) >= 2 else price
        tech["pct_change"] = round((price / prev - 1) * 100, 2) if prev else 0.0
        tech["ret_60d"] = round((price / float(close.iloc[-60]) - 1) * 100, 2) if len(close) >= 60 else 0.0

        mas = [tech.get(f"ma{n}") for n in [20, 60, 120, 200]]
        tech["is_uptrend"] = all(a is not None and b is not None and a > b for a, b in zip(mas, mas[1:]))
        return tech
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────
#  3. 스코어 계산 
# ─────────────────────────────────────────────────────────────
def score_insider_gov(insider_df, market_cap):
    base = 5.0
    if insider_df is None or insider_df.empty: return round(base, 2)
    try:
        df = insider_df.copy()
        def find_col(keywords):
            for kw in keywords:
                for c in df.columns:
                    if kw.lower() in str(c).lower(): return c
            return None
        date_c, type_c, role_c, val_c = find_col(["date", "startDate"]), find_col(["transaction", "description"]), find_col(["title", "position", "relation"]), find_col(["value", "money"])

        if date_c:
            try:
                col = pd.to_datetime(df[date_c], errors="coerce")
                if col.dt.tz is not None: col, cutoff = col.dt.tz_convert("UTC"), pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=INSIDER_DAYS)
                else: cutoff = pd.Timestamp.now() - pd.Timedelta(days=INSIDER_DAYS)
                df = df[col >= cutoff]
            except: pass

        if df.empty or type_c is None: return round(base, 2)
        type_s = df[type_c].astype(str).str.lower()
        buys, sells = df[type_s.str.contains("buy|purchase", na=False, regex=True)], df[type_s.str.contains("sale|sell", na=False, regex=True)]
        
        net = len(buys) - len(sells) * 0.5
        if net >= 3: base += 1.5
        elif net >= 2: base += 1.0
        elif net >= 1: base += 0.5
        elif net <= -3: base -= 2.0
        elif net <= -2: base -= 1.2
        elif net < 0: base -= 0.5

        KEY_ROLES = ["ceo", "cfo", "chief executive", "chief financial", "president"]
        if role_c:
            if not buys.empty and buys[role_c].astype(str).str.lower().str.contains("|".join(KEY_ROLES), na=False, regex=True).any(): base += 1.0
            if not sells.empty and sells[role_c].astype(str).str.lower().str.contains("|".join(KEY_ROLES), na=False, regex=True).any(): base -= 1.0

        if val_c and market_cap and market_cap > 0:
            try:
                buy_v, sell_v = pd.to_numeric(buys[val_c], errors="coerce").fillna(0).sum(), pd.to_numeric(sells[val_c], errors="coerce").fillna(0).sum()
                ratio = (buy_v - sell_v) / market_cap
                if ratio > 0.001: base += 1.0
                elif ratio > 0.0001: base += 0.5
                elif ratio < -0.001: base -= 1.5
                elif ratio < -0.0001: base -= 0.8
            except: pass
    except: pass
    return max(0.0, min(10.0, round(base, 2)))

def calc_market_env_score(env_data):
    score = 7.5
    if "^VIX" in env_data:
        try:
            vix = float(env_data["^VIX"]["Close"].iloc[-1])
            if vix < 15: score += 4.0
            elif vix < 20: score += 2.5
            elif vix < 25: score += 0.5
            elif vix < 30: score -= 2.0
            else: score -= 4.0
        except: pass
    if "QQQ" in env_data:
        try:
            qqq = env_data["QQQ"]["Close"].dropna()
            cur = float(qqq.iloc[-1])
            if len(qqq) >= 200:
                r = cur / float(qqq.rolling(200).mean().iloc[-1])
                if r > 1.05: score += 2.5
                elif r > 1.0: score += 1.0
                elif r > 0.97: score -= 1.0
                else: score -= 2.5
        except: pass
    return max(0.0, min(15.0, round(score, 2)))

def score_fundamental(fund, tech):
    score = 10.0
    eg = fund.get("earnings_growth")
    if eg is not None:
        if   eg > 0.30: score += 10.0  
        elif eg > 0.15: score += 6.0   
        elif eg > 0.05: score += 2.5
        elif eg > 0.0:  score += 0.5
        else:           score -= 3.0

    rg = fund.get("revenue_growth")
    if rg is not None:
        if   rg > 0.20: score += 6.0   
        elif rg > 0.10: score += 3.0   
        elif rg > 0.0:  score += 0.5
        else:           score -= 1.5

    pm = fund.get("profit_margin")
    if pm is not None:
        if   pm > 0.25: score += 3.0
        elif pm > 0.15: score += 1.5
        elif pm > 0.05: score += 0.5
        elif pm <  0.0: score -= 2.0

    fcf  = fund.get("fcf")
    mcap = fund.get("market_cap") or 0
    if fcf is not None and mcap > 0:
        fcf_yield = fcf / mcap
        if   fcf_yield > 0.04: score += 3.0  
        elif fcf_yield > 0.02: score += 1.5  
        elif fcf_yield < 0:    score -= 1.5  

    roe = fund.get("roe")
    if roe is not None:
        if   roe > 0.25: score += 2.0  
        elif roe > 0.15: score += 1.0  
        elif roe <  0.0: score -= 1.5  

    target = fund.get("analyst_target")
    cur    = fund.get("current_price") or tech.get("price")
    if target and cur and cur > 0:
        upside = (target / cur - 1) * 100
        if   upside > 30: score += 4.0
        elif upside > 15: score += 2.0
        elif upside >  5: score += 0.5
        elif upside < -5: score -= 2.0
    return max(0.0, min(30.0, round(score, 2)))

def score_macro(ticker, fund):
    score = 10.0
    mcap = fund.get("market_cap", 0) or 0
    if   mcap > 1e12:  score += 3.0  
    elif mcap > 2e11:  score += 1.5  
    elif mcap < 1e10:  score -= 1.0  

    sec = fund.get("sector", "")
    if   sec in ("Technology", "Communication Services"): score += 1.5
    elif sec in ("Consumer Defensive", "Utilities", "Real Estate"): score -= 1.5

    peg = fund.get("peg_ratio")
    if peg is not None and peg > 0:
        if   peg > 5.0:  score -= 6.0   
        elif peg > 3.0:  score -= 3.0   
        elif peg < 1.0:  score += 6.0   
        elif peg < 1.5:  score += 4.0   
        elif peg < 2.5:  score += 1.5   
    return max(0.0, min(25.0, round(score, 2)))

def score_sentiment(fund, tech):
    score = 5.0
    vr = tech.get("vol_ratio", 1.0)
    if   vr < 0.6:  score += 2.5   
    elif vr < 0.9:  score += 1.0
    elif vr < 1.5:  score += 0.0   
    elif vr < 2.5:  score -= 1.5   
    else:           score -= 3.5   

    high52 = fund.get("52w_high")
    price  = tech.get("price")
    if high52 and price and high52 > 0:
        ratio = price / high52
        if   ratio > 0.97: score += 1.5   
        elif ratio > 0.85: score += 0.5
        elif ratio < 0.70: score -= 1.5   

    inst = fund.get("inst_pct")
    if inst is not None:
        if   inst > 0.80: score += 2.0   
        elif inst > 0.60: score += 1.0
        elif inst < 0.30: score -= 1.0   

    short = fund.get("short_pct")
    if short is not None:
        if   short < 0.02: score += 2.0   
        elif short < 0.05: score += 0.5
        elif short > 0.15: score -= 2.0   
        elif short > 0.10: score -= 1.0
    return max(0.0, min(10.0, round(score, 2)))

def score_technical(tech):
    score = 8.0
    price, ma60, ma200 = tech.get("price", 0), tech.get("ma60"), tech.get("ma200")
    rsi, macd_diff, vr = tech.get("rsi", 50), tech.get("macd_diff", 0), tech.get("vol_ratio", 1.0)
    bb_lower, bb_upper, pct = tech.get("bb_lower"), tech.get("bb_upper"), tech.get("pct_change", 0)

    if ma200 and price < ma200 * 0.97 and pct < -4 and vr > 2.0: return 0.0, True

    if ma200:
        ratio = price / ma200
        if ratio > 1.10: score += 4.0
        elif ratio > 1.03: score += 2.0
        elif ratio > 0.97: score += 0.0
        elif ratio > 0.93: score -= 3.0
        else: score -= 5.0

    if ma60 and price:
        if price > ma60 and price < ma60 * 1.05: score += 3.0
        elif price < ma60 * 0.95: score -= 2.0

    if macd_diff > 0: score += 1.5
    elif macd_diff < -0.5: score -= 1.5

    if rsi < 30: score += 3.0
    elif rsi < 45: score += 1.0
    elif rsi > 80: score -= 4.0
    elif rsi > 70: score -= 2.0

    if tech.get("is_uptrend", False): score += 2.0

    if bb_lower and bb_upper and price:
        bb_pos = (price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
        if bb_pos < 0.2: score += 2.0
        elif bb_pos > 0.8: score -= 1.5
    return max(0.0, min(20.0, round(score, 2))), False

def calc_total_score(ticker, fund, tech, market_env_score, insider_df=None):
    p1 = score_fundamental(fund, tech)
    p2 = score_macro(ticker, fund)
    p3 = score_sentiment(fund, tech)
    p4 = market_env_score
    p5_val, zero_flag = score_technical(tech)
    p6_val = score_insider_gov(insider_df, fund.get("market_cap"))
    
    bb_upper = tech.get("bb_upper")
    bb_lower = tech.get("bb_lower")
    price = tech.get("price", 0)
    bb_pos = (price - bb_lower) / (bb_upper - bb_lower) if (bb_upper and bb_lower and bb_upper != bb_lower) else None
    fcf = fund.get("fcf")
    mcap = fund.get("market_cap")
    fcf_yield = round(fcf / mcap, 4) if (fcf and mcap) else None

    return {
        "ticker": ticker, "name": fund.get("short_name", ticker), "price": price, "pct_change": tech.get("pct_change", 0),
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5_val, "p6": p6_val,
        "total": round(p1 + p2 + p3 + p4 + p5_val + p6_val, 2), "zero_flag": zero_flag, 
        "rsi": tech.get("rsi", 50), "vol_ratio": tech.get("vol_ratio", 1.0), 
        "ma20": tech.get("ma20"), "ma60": tech.get("ma60"), "ma120": tech.get("ma120"), "ma200": tech.get("ma200"),
        "bb_position": round(bb_pos, 4) if bb_pos is not None else None,
        "fcf_yield": fcf_yield, "ret_60d": tech.get("ret_60d", 0)
    }

def determine_signal(row):
    total = row["total"]
    p3    = row["p3"]
    rsi   = row["rsi"]
    vr    = row["vol_ratio"]
    price = row["price"]
    ma120 = row.get("ma120")

    if row["zero_flag"]: return "STRONG_SELL", "🔴🔴 즉시 청산"
    if total < 50:
        if ma120 and price < ma120: return "SELL", "🔴 전량 손절"
        else: return "PARTIAL_SELL", "🟡 비중 축소 (추세 유지)"
    if p3 < 4.0 and rsi > 72: return "PARTIAL_SELL", "🟡 25% 익절"
    if vr > 2.5 and rsi > 70: return "PARTIAL_SELL", "🟡 과열 익절"
    
    if   total >= 95: return "APEX",        "🏆 APEX 최우선 매수"
    elif total >= 83: return "STRONG_BUY",  "🟢🟢 강력 매수"
    elif total >= 72: return "BUY",         "🟢 매수"
    elif total >= 60: return "HOLD",        "🔵 보유/관망"
    else:             return "WEAK",        "⚪ 주의 관찰"

def calc_allocation(ranked_df, total_budget=55):
    valid_df = ranked_df[(ranked_df["total"] >= 60) & (~ranked_df["zero_flag"])].copy()
    top10 = valid_df.head(10).copy()
    if top10.empty: return top10
    n = len(top10)
    weights = np.array([(n + 1 - i) ** 1.5 for i in range(1, n + 1)])
    weights /= weights.sum()
    top10["allocation"] = (weights * total_budget).round(2)
    return top10

# ─────────────────────────────────────────────────────────────
#  4. HTML 생성
# ─────────────────────────────────────────────────────────────
def get_score_color(total):
    if   total >= 95: return "#7c3aed"
    elif total >= 83: return "#16a34a"
    elif total >= 72: return "#22c55e"
    elif total >= 60: return "#3b82f6"
    elif total >= 50: return "#f59e0b"
    else:             return "#dc2626"

def get_sig_class(signal):
    return {
        "APEX": "apex", "STRONG_BUY": "strong-buy", "BUY": "buy",
        "HOLD": "hold", "PARTIAL_SELL": "partial", "SELL": "sell",
        "STRONG_SELL": "strong-sell", "WEAK": "weak",
    }.get(signal, "weak")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>퀀트멘털 v8.9 일일 리포트 — {{ date }}</title>
<style>
  :root { --blue: #2563eb; --green: #16a34a; --red: #dc2626; --yellow: #d97706; --gray: #64748b; --bg: #f8fafc; }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif; background: var(--bg); color: #1e293b; font-size: 13px; }
  .header { background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 60%, #2563eb 100%); color: #fff; padding: 28px 32px 20px; }
  .header-sub  { font-size: 10px; letter-spacing: 2px; opacity: .6; margin-bottom: 6px; }
  .header-title{ font-size: 26px; font-weight: 800; margin-bottom: 4px; }
  .header-meta { font-size: 12px; opacity: .75; }
  .header-stats{ display:flex; gap:24px; margin-top:14px; }
  .stat-box { background: rgba(255,255,255,.1); border-radius:8px; padding: 8px 16px; text-align:center; }
  .stat-val { font-size:18px; font-weight:800; }
  .stat-lbl { font-size:10px; opacity:.7; margin-top:2px; }
  .market-bar { background: #1e293b; color: #e2e8f0; padding: 10px 32px; display:flex; gap:24px; flex-wrap:wrap; font-size: 12px; }
  .mkt-item { display:flex; flex-direction:column; align-items:center; }
  .mkt-name { font-size:10px; color:#94a3b8; }
  .mkt-val  { font-weight:700; font-size:13px; }
  .up   { color: #4ade80; }
  .down { color: #f87171; }
  .section { padding: 20px 32px; }
  .sec-title { font-size:16px; font-weight:800; margin-bottom:14px; border-left: 4px solid var(--blue); padding-left: 10px; }
  
  /* 테이블 디자인 */
  .rank-table { width:100%; border-collapse:collapse; }
  .rank-table th { background: #1e293b; color: #e2e8f0; padding: 9px 8px; text-align:center; font-size:11px; font-weight:700; white-space:nowrap; }
  .rank-table td { padding: 8px 8px; text-align:center; border-bottom: 1px solid #e2e8f0; font-size:12px; white-space:nowrap; }
  .rank-table tr:hover td { background: #eff6ff; }
  .rank-num  { font-weight:800; font-size:15px; color: var(--blue); }
  .ticker    { font-weight:800; font-size:13px; }
  .total-score{ font-weight:800; font-size:14px; }
  .score-bar { height:6px; border-radius:3px; margin:2px auto; background: linear-gradient(90deg, #3b82f6, #8b5cf6); }
  .alloc-badge { background: #eff6ff; border: 1px solid #bfdbfe; border-radius:6px; padding:2px 8px; font-weight:700; color: var(--blue); font-size:12px; }
  .sig-apex        { background:#ede9fe; color:#5b21b6; border:1px solid #c4b5fd; border-radius:20px; padding:3px 10px; font-weight:800; font-size:11px; }
  .sig-strong-buy  { background:#dcfce7; color:#166534; border:1px solid #86efac; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-buy         { background:#f0fdf4; color:#16a34a; border:1px solid #bbf7d0; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-hold        { background:#eff6ff; color:#1d4ed8; border:1px solid #bfdbfe; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-partial     { background:#fffbeb; color:#92400e; border:1px solid #fde68a; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-sell        { background:#fee2e2; color:#991b1b; border:1px solid #fca5a5; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-strong-sell { background:#7f1d1d; color:#fee2e2; border:1px solid #dc2626; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .sig-weak        { background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1; border-radius:20px; padding:3px 10px; font-weight:700; font-size:11px; }
  .alert-box { background:#fef2f2; border:1px solid #fecaca; border-radius:10px; padding:14px 18px; margin-bottom:16px; }
  .alert-title { font-weight:800; color:#dc2626; margin-bottom:6px; font-size:14px; }
  .alert-item { font-size:12px; color:#7f1d1d; line-height:1.8; }
  .buy-box { background:#f0fdf4; border:1px solid #bbf7d0; border-radius:10px; padding:14px 18px; margin-bottom:16px; }
  .buy-title { font-weight:800; color:#16a34a; margin-bottom:6px; font-size:14px; }
  
  /* 클로드 스타일 브리핑 패널 CSS */
  .panel{background:#fff;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.05);}
  .ph{padding:14px 20px;background:#f8fafc;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;}
  .pt{font-size:13px;font-weight:800;letter-spacing:0.05em;color:#1e293b;}
  .pb-badge {font-size:10px;padding:3px 8px;border-radius:12px;font-weight:700;}
  .pb-cyan {color:#0369a1;background:#e0f2fe;border:1px solid #bae6fd;}
  .pb-amber {color:#b45309;background:#fef3c7;border:1px solid #fde68a;}
  .hero-box {background: linear-gradient(145deg, #0f172a 0%, #1e3a5f 100%); color: #e2e8f0; padding: 24px 32px; border-radius: 8px; border-left: 4px solid var(--blue); margin-bottom: 24px;}
  .hero-title {font-size: 22px; font-weight: 800; color: #fff; margin-bottom: 12px; line-height: 1.3;}
  .tl-item{display:grid;grid-template-columns:88px 1fr;gap:16px;padding:16px 20px;border-bottom:1px solid #e2e8f0;}
  .tl-item:last-child{border-bottom:none;}
  .tl-date{font-size:12px;font-weight:800;color:#64748b;}
  .tl-head{font-size:14px;font-weight:700;color:#0f172a;margin-bottom:4px;}
  .mn-item{display:grid;grid-template-columns:60px 1fr;gap:16px;padding:16px 20px;border-bottom:1px solid #e2e8f0;}
  .mn-tick{font-size:13px;font-weight:800;color:#2563eb;}
  .exp-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;}
  .exp-item{padding:20px;border-right:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;}
  .exp-item:nth-child(2n){border-right:none;}
  .exp-name{font-size:12px;font-weight:800;color:#0f172a;margin-bottom:8px;}
  .beg-grid{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:20px;}
  .beg-head{font-size:12px;font-weight:800;color:#2563eb;margin-bottom:8px;}
  .pos {color:#16a34a;font-weight:700;}
  .neg {color:#dc2626;font-weight:700;}

  .footer { background:#1e293b; color:#94a3b8; text-align:center; padding:16px; font-size:11px; line-height:1.8; }
</style>
</head>
<body>
<div class="header">
  <div class="header-sub">QUANTAMENTAL v8.9 · REAL-TIME AI ENGINE · NASDAQ 100</div>
  <div class="header-title">📊 퀀트멘털 일일 리포트</div>
  <div class="header-meta">분석 완료: {{ current_time }}<br>데이터 기준일 (미국 증시 종가): <strong style="color:#dcfce7">{{ date }}</strong> · 분석 종목: {{ total_analyzed }}개</div>
  <div class="header-stats">
    <div class="stat-box"><div class="stat-val">{{ strong_buy_count }}</div><div class="stat-lbl">강력매수</div></div>
    <div class="stat-box"><div class="stat-val">{{ buy_count }}</div><div class="stat-lbl">매수</div></div>
    <div class="stat-box"><div class="stat-val">{{ sell_count }}</div><div class="stat-lbl">매도/청산</div></div>
    <div class="stat-box"><div class="stat-val">{{ avg_score }}</div><div class="stat-lbl">평균점수</div></div>
  </div>
</div>
<div class="market-bar">
  {% for item in market_items %}
  <div class="mkt-item"><span class="mkt-name">{{ item.name }}</span><span class="mkt-val {{ item.cls }}">{{ item.value }}</span></div>
  {% endfor %}
</div>

{% if sell_alerts %}
<div class="section"><div class="alert-box"><div class="alert-title">⚠️ 매도 / 청산 긴급 신호</div>
{% for a in sell_alerts %}<div class="alert-item">{{ a }}</div>{% endfor %}
</div></div>
{% endif %}

{% if buy_highlights %}
<div class="section" style="padding-top:0"><div class="buy-box"><div class="buy-title">💰 오늘의 매수 추천 (자금 배분 포함)</div>
{% for b in buy_highlights %}<div class="alert-item">{{ b }}</div>{% endfor %}
</div></div>
{% endif %}

<div class="section" style="padding-top:0">
  <div class="sec-title">📋 나스닥 100 퀀트멘털 전체 순위</div>
  <div style="overflow-x:auto">
  <table class="rank-table">
    <thead>
      <tr>
        <th>순위</th><th>종목</th><th>현재가</th><th>등락률</th><th>종합점수</th>
        <th title="펀더멘털">①기본</th><th title="매크로">②매크로</th><th title="센티먼트">③심리</th>
        <th title="시장환경">④시장</th><th title="기술적">⑤기술</th><th title="내부자">⑥내부자</th>
        <th>RSI</th><th>거래량비율</th><th>배분금액</th><th>신호</th>
      </tr>
    </thead>
    <tbody>
    {% for row in rows %}
      <tr>
        <td><span class="rank-num">{{ row.rank }}</span></td>
        <td><div class="ticker">{{ row.ticker }}</div><div style="font-size:10px;color:#64748b">{{ row.name[:15] }}</div></td>
        <td>${{ "%.2f"|format(row.price) }}</td>
        <td class="{{ 'up' if row.pct_change >= 0 else 'down' }}">{{ "%+.2f"|format(row.pct_change) }}%</td>
        <td><span class="total-score" style="color:{{ row.score_color }}">{{ "%.1f"|format(row.total) }}</span><div class="score-bar" style="width:{{ (row.total/110*80)|int }}px"></div></td>
        <td>{{ "%.1f"|format(row.p1) }}<span style="color:#94a3b8">/30</span></td>
        <td>{{ "%.1f"|format(row.p2) }}<span style="color:#94a3b8">/25</span></td>
        <td>{{ "%.1f"|format(row.p3) }}<span style="color:#94a3b8">/10</span></td>
        <td>{{ "%.1f"|format(row.p4) }}<span style="color:#94a3b8">/15</span></td>
        <td>{{ "%.1f"|format(row.p5) }}<span style="color:#94a3b8">/20</span></td>
        <td style="font-weight:bold;color:#2563eb">{{ "%.1f"|format(row.p6) }}<span style="color:#94a3b8;font-weight:normal">/10</span></td>
        <td class="{{ 'down' if row.rsi > 70 else ('up' if row.rsi < 35 else '') }}">{{ "%.0f"|format(row.rsi) }}</td>
        <td class="{{ 'down' if row.vol_ratio > 2.0 else '' }}">{{ "%.1f"|format(row.vol_ratio) }}x</td>
        <td>{% if row.allocation is not none %}<span class="alloc-badge">${{ "%.1f"|format(row.allocation) }}</span>{% else %}<span style="color:#94a3b8">—</span>{% endif %}</td>
        <td><span class="sig-{{ row.sig_class }}">{{ row.sig_label }}</span></td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>

{% if top20_summaries and top20_summaries.get('hero_title') %}
<div class="section">
  <div class="sec-title">🎯 마켓 인텔리전스 (AI 통합 브리핑)</div>
  
  <div class="hero-box">
    <div class="hero-title">{{ top20_summaries.hero_title }}</div>
    <div style="font-size:13px;line-height:1.7;">{{ top20_summaries.hero_body | safe }}</div>
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
    <div class="panel">
      <div class="ph"><span class="pt">주요 이슈 타임라인</span><span class="pb-badge pb-cyan">MARKET FLOW</span></div>
      <div>
        {% for item in top20_summaries.timeline %}
        <div class="tl-item">
          <div class="tl-date">{{ item.date }}<br><span style="font-size:10px;font-weight:normal;">{{ item.trend }}</span></div>
          <div>
            <div class="tl-head">{{ item.title | safe }}</div>
            <div style="font-size:12px;color:#475569;line-height:1.6;">{{ item.desc | safe }}</div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>

    <div class="panel">
      <div class="ph"><span class="pt">모닝 노트 (특징주 뉴스)</span><span class="pb-badge pb-amber">MORNING NOTE</span></div>
      <div>
        {% for note in top20_summaries.morning_notes %}
        <div class="mn-item">
          <div class="mn-tick">{{ note.ticker }}</div>
          <div>
            <div class="tl-head">{{ note.title | safe }}</div>
            <div style="font-size:12px;color:#475569;line-height:1.6;">{{ note.desc | safe }}</div>
          </div>
        </div>
        {% endfor %}
      </div>
    </div>
  </div>

  <div class="panel">
    <div class="ph"><span class="pt">전문가 나침반 (시장 뷰)</span><span class="pb-badge pb-cyan">EXPERT COMPASS</span></div>
    <div class="exp-grid">
      {% for exp in top20_summaries.expert_views %}
      <div class="exp-item">
        <div class="exp-name">{{ exp.name }}</div>
        <div style="font-size:12px;color:#475569;line-height:1.7;">{{ exp.desc | safe }}</div>
      </div>
      {% endfor %}
    </div>
  </div>

  <div class="panel">
    <div class="ph"><span class="pt">💡 초보 투자자 핵심 정리</span><span class="pb-badge pb-amber">BEGINNER GUIDE</span></div>
    <div class="beg-grid">
      <div>
        <div class="beg-head">오늘 장세 한 문장 요약</div>
        <div style="font-size:12px;color:#475569;line-height:1.7;">{{ top20_summaries.beginner_guide.summary | safe }}</div>
      </div>
      <div>
        <div class="beg-head">지금 가져야 할 마음가짐 & 전략</div>
        <div style="font-size:12px;color:#475569;line-height:1.7;">{{ top20_summaries.beginner_guide.mindset | safe }}</div>
      </div>
    </div>
  </div>
</div>
{% endif %}

<div class="section">
  <div class="sec-title">📌 점수 기준 안내 (수정 전략 v8.6)</div>
  <table style="width:100%;border-collapse:collapse;font-size:12px">
    <tr style="background:#f1f5f9">
      <th style="padding:8px;text-align:left">기둥</th><th style="padding:8px;text-align:center">만점</th><th style="padding:8px;text-align:left">고득점 기준</th><th style="padding:8px;text-align:left">저득점 기준</th>
    </tr>
    {% for g in guide_rows %}
    <tr style="border-bottom:1px solid #e2e8f0">
      <td style="padding:8px;font-weight:700">{{ g.name }}</td><td style="padding:8px;text-align:center;font-weight:700;color:#2563eb">{{ g.max }}</td><td style="padding:8px;color:#166534">{{ g.high }}</td><td style="padding:8px;color:#dc2626">{{ g.low }}</td>
    </tr>
    {% endfor %}
  </table>
</div>
<div class="footer">
  퀀트멘털 v8.9 실시간 AI 자동 생성 리포트 · 데이터 기준일: {{ date }}<br>
  ⚠️ 본 리포트는 투자 참고용이며, 모든 투자 결정과 손익의 책임은 투자자 본인에게 있습니다.
</div>
</body>
</html>
"""

def generate_report(results_df, market_env_data, market_date, top20_summaries=None, output_path="quantamental_report.html"):
    current_time_str = datetime.now().strftime("%Y년 %m월 %d일 %H:%M")
    top10_alloc = calc_allocation(results_df)
    alloc_map = dict(zip(top10_alloc["ticker"], top10_alloc["allocation"]))

    market_items = []
    for sym, label in [("QQQ","QQQ"),("^VIX","VIX"),("SPY","SPY"),("^IXIC","NASDAQ"),("^TNX","US 10Y")]:
        if sym in market_env_data:
            df = market_env_data[sym]
            try:
                cur = float(df["Close"].iloc[-1])
                prev= float(df["Close"].iloc[-2]) if len(df)>1 else cur
                chg = (cur/prev - 1)*100
                cls = "up" if chg >= 0 else "down"
                if sym == "^VIX": val_str = f"{cur:.2f} ({chg:+.2f}%)"
                elif sym == "^TNX": val_str = f"{cur:.3f}% ({chg:+.2f}%)"
                else: val_str = f"${cur:,.2f} ({chg:+.2f}%)"
                market_items.append({"name": label, "value": val_str, "cls": cls})
            except: pass

    sell_alerts, buy_highlights = [], []
    for _, r in results_df.iterrows():
        sig, label = determine_signal(r)
        if sig in ["STRONG_SELL","SELL"]:
            sell_alerts.append(f"🔴 {r['ticker']} ({str(r.get('name', r['ticker']))[:15]}) — 총점 {r['total']:.1f}점 / {label}")

    for i, (_, r) in enumerate(top10_alloc.iterrows(), 1):
        sig, label = determine_signal(r)
        buy_highlights.append(f"[{i:2d}위] {r['ticker']:6s} 총점 {r['total']:.1f}점 → 배분 <strong>${r['allocation']:.1f}</strong> / {label}")

    rows = []
    for rank, (_, r) in enumerate(results_df.iterrows(), 1):
        sig, label = determine_signal(r)
        rows.append({
            "rank": rank, "ticker": r["ticker"], "name": r.get("name", r["ticker"]),
            "price": r["price"], "pct_change": r["pct_change"], "total": r["total"],
            "p1": r["p1"], "p2": r["p2"], "p3": r["p3"], "p4": r["p4"],
            "p5": r["p5"], "p6": r.get("p6", 5.0),
            "rsi": r["rsi"], "vol_ratio": r["vol_ratio"], "score_color": get_score_color(r["total"]),
            "sig_class": get_sig_class(sig), "sig_label": label, "allocation": alloc_map.get(r["ticker"], None),
        })

    guide_rows = [
        {"name":"① 펀더멘털",  "max":"30","high":"EPS↑ 매출↑ FCF↑ ROE15%+ 순이익률↑","low":"EPS하향·역성장·FCF음수·ROE마이너스"},
        {"name":"② 매크로/가치", "max":"25","high":"PEG 1.0 이하 저평가·초대형주","low":"PEG 3.0 이상 고평가·소형주"},
        {"name":"③ 센티먼트/수급", "max":"10","high":"거래량 조용·52주 신고가 근접","low":"거래량 폭발·고점 대비 대폭 하락"},
        {"name":"④ 시장환경",      "max":"15","high":"VIX<15·QQQ 200일선 위","low":"VIX>30·QQQ 200일선 붕괴"},
        {"name":"⑤ 기술적 타점",   "max":"20","high":"눌림목·정배열·RSI 반등·MACD↑","low":"200일선 붕괴·RSI 과열·0점(매도)"},
        {"name":"⑥ 내부자 거래",   "max":"10","high":"CEO/CFO 직접 매수·다수 동시매수·시총0.1%+","low":"클러스터 매도·CEO 대량매도"},
    ]

    html = Template(HTML_TEMPLATE).render(
        date=market_date, current_time=current_time_str, total_analyzed=len(results_df),
        strong_buy_count=sum(1 for _, r in results_df.iterrows() if r["total"] >= 83),
        buy_count=sum(1 for _, r in results_df.iterrows() if 72 <= r["total"] < 83),
        sell_count=sum(1 for _, r in results_df.iterrows() if r["total"] < 50 or r["zero_flag"]),
        avg_score=f"{results_df['total'].mean():.1f}",
        market_items=market_items, sell_alerts=sell_alerts,
        buy_highlights=buy_highlights, rows=rows, guide_rows=guide_rows,
        top20_summaries=top20_summaries
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path

# ─────────────────────────────────────────────────────────────
#  5. 메인 실행 블록
# ─────────────────────────────────────────────────────────────
def run(tickers=None, output=None):
    if tickers is None: tickers = get_latest_nasdaq100()
    if output is None:
        # 💡 다운로드 폴더가 아닌 reports 폴더에 자동 저장하도록 변경 (GitHub Actions 구조 유지)
        os.makedirs("reports", exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d_%H%M")
        output = f"reports/quantamental_report_{date_str}.html"

    print(f"\n📊 퀀트멘털 v8.9 엔진 시작! 총 {len(tickers)}개 종목 분석\n") 
    all_stock_data, all_fundamentals, all_insider = {}, {}, {}
    
    for i in range(0, len(tickers), BATCH_SIZE):
        chunk = tickers[i:i + BATCH_SIZE]
        print(f"▶ 데이터 수집 중... ({chunk[0]}~{chunk[-1]})")
        
        chunk_stock_data = fetch_price_batch(chunk)
        valid = list(chunk_stock_data.keys())
        
        if valid:
            chunk_fundamentals = fetch_fundamentals(valid)
            chunk_insider, _ = fetch_insider_and_gov(valid)
        else:
            chunk_fundamentals, chunk_insider = {}, {}
            
        all_stock_data.update(chunk_stock_data)
        all_fundamentals.update(chunk_fundamentals)
        all_insider.update(chunk_insider) 
        
    print("\n✅ 시장 환경 데이터 수집 및 점수 계산...")
    market_env = fetch_market_env()
    env_score  = calc_market_env_score(market_env)

    try:
        if "QQQ" in market_env and not market_env["QQQ"].empty:
            market_date = market_env["QQQ"].index[-1].strftime("%Y-%m-%d")
        else:
            market_date = datetime.now().strftime("%Y-%m-%d")
    except:
        market_date = datetime.now().strftime("%Y-%m-%d")

    all_scores = []
    for tkr in all_stock_data.keys():
        try:
            tech = calc_technicals(all_stock_data[tkr])
            if not tech: continue
            fund = all_fundamentals.get(tkr, {})
            insider = all_insider.get(tkr, None)
            score_row = calc_total_score(tkr, fund, tech, env_score, insider)
            all_scores.append(score_row)
        except Exception as e:
            print(f"  ⚠️ {tkr} 스코어 계산 오류: {e}")

    if not all_scores:
        print("\n  ❌ 데이터를 성공적으로 계산한 종목이 하나도 없습니다.")
        return None, None

    results_df = pd.DataFrame(all_scores).sort_values("total", ascending=False).reset_index(drop=True)
    
    # ─────────────────────────────────────────────────────────
    # AI 학습을 위한 시계열 누적 데이터 저장 (시장 날짜 기준)
    # ─────────────────────────────────────────────────────────
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    except Exception:
        script_dir = os.getcwd()
        
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    master_csv_path = os.path.join(data_dir, "nasdaq100data.csv")
    master_parquet_path = os.path.join(data_dir, "nasdaq100data.parquet")
    
    ai_df = results_df.copy()
    ai_df.insert(0, 'date', market_date) 
    
    vix_val = float(market_env["^VIX"]["Close"].iloc[-1]) if "^VIX" in market_env and not market_env["^VIX"].empty else None
    tnx_val = float(market_env["^TNX"]["Close"].iloc[-1]) if "^TNX" in market_env and not market_env["^TNX"].empty else None
    
    ai_df.insert(1, 'vix', vix_val)
    ai_df.insert(2, 'us10y_yield', tnx_val)
    ai_df['signal'] = ai_df.apply(lambda r: determine_signal(r)[0], axis=1)
    
    skip_save = False
    try:
        if os.path.exists(master_csv_path):
            existing = pd.read_csv(master_csv_path, encoding="utf-8-sig")
            if "date" in existing.columns:
                existing["date"] = pd.to_datetime(existing["date"]).dt.strftime("%Y-%m-%d")
                if market_date in existing["date"].values:
                    skip_save = True
                    
        if not skip_save:
            ai_df.to_csv(master_csv_path, mode='a' if os.path.exists(master_csv_path) else 'w', 
                         header=not os.path.exists(master_csv_path), index=False, encoding="utf-8-sig")
            try:
                if os.path.exists(master_parquet_path):
                    existing_pq = pd.read_parquet(master_parquet_path)
                    combined_pq = pd.concat([existing_pq, ai_df], ignore_index=True)
                    combined_pq.to_parquet(master_parquet_path, engine="pyarrow")
                else:
                    ai_df.to_parquet(master_parquet_path, engine="pyarrow")
            except Exception as e:
                pass
    except Exception as e:
        pass
        
    # ── 🌟 실시간 최신 마켓 브리핑 엔진 (Claude 스타일 + JSON 프롬프팅) ──────────
    print("\n🌐 시장 전체 및 주요 종목 뉴스 수집 중 (RSS)...")
    
    target_tickers = ["SPY", "QQQ"] + [r["ticker"] for _, r in results_df.head(7).iterrows()]
    aggregated_news = []

    for tkr in target_tickers:
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={tkr}&region=US&lang=en-US"
            # ⭐️ 깃허브 서버 IP 차단 우회를 위한 브라우저 헤더 위장
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            req = urllib.request.Request(url, headers=headers)
            xml_data = urllib.request.urlopen(req, timeout=5).read()
            root = ET.fromstring(xml_data)
            
            count = 0
            for item in root.findall('.//item'):
                title = item.find('title').text if item.find('title') is not None else ""
                pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ""
                if title:
                    aggregated_news.append(f"[{tkr}] {title} ({pub_date[:16]})")
                    count += 1
                if count >= 4: break
        except Exception as e:
            print(f"  [!] {tkr} 뉴스 수집 실패: {e}")
            pass
        time.sleep(0.3)

    # ⭐️ 야후에서 완전히 차단당해 뉴스를 못 가져왔을 때의 안전장치
    if not aggregated_news:
        print("⚠️ 주의: 통신 제한으로 뉴스를 가져오지 못했습니다. 기술적 지표 기반으로 브리핑을 대체합니다.")
        aggregated_news.append("[시장 알림] 현재 외부 뉴스 서버 접속 제한으로 세부 뉴스를 불러올 수 없습니다. 오늘 수집된 주요 기술적 분석 지표와 증시 전반의 흐름만으로 시황을 유추하여 브리핑을 작성하세요.")

    news_text_block = "\n".join(aggregated_news)

    print("🤖 Gemini: 수집된 전체 뉴스를 바탕으로 클로드 수준의 심층 브리핑 리포트를 작성합니다...")
    
    prompt = f"""
    당신은 월스트리트 최고 수준의 시황 분석가입니다. 
    다음은 오늘 시장 대표 ETF(SPY, QQQ) 및 나스닥 핵심 종목들의 최신 뉴스 헤드라인입니다.
    이 데이터를 종합하여, 지정된 JSON 형식으로 전문가 수준의 데일리 마켓 브리핑을 작성해 주세요.
    한국어로 작성하며, 어조는 단호하고 명확하게 해주세요. 긍정적인 부분은 <span class="pos">, 부정적인 부분은 <span class="neg">, 강조할 부분은 <strong> 태그를 활용해도 좋습니다.

    [입력 데이터]
    {news_text_block}

    [출력 조건]
    반드시 마크다운이나 기타 텍스트 없이 순수한 JSON 객체만 출력하세요. 구조는 완벽히 아래와 같아야 합니다:
    {{
      "hero_title": "오늘 시장을 요약하는 강력한 헤드라인",
      "hero_body": "간밤의 시장 흐름과 오늘 주목할 핵심 변수를 3~4문장으로 깊이 있게 요약",
      "timeline": [
        {{"date": "최근 이슈1", "trend": "상승/하락/혼조", "title": "이슈 요약", "desc": "상세 내용"}},
        {{"date": "오늘 예정", "trend": "주목", "title": "오늘 주요 이벤트", "desc": "상세 내용"}}
      ],
      "morning_notes": [
        {{"ticker": "종목 티커", "title": "뉴스 핵심 제목", "desc": "상세 설명 (최대 6개)"}}
      ],
      "expert_views": [
        {{"name": "월가 주요 기관 시각", "desc": "시장 상황에 대한 평가 (최대 4개)"}}
      ],
      "beginner_guide": {{
        "summary": "오늘 장세 초보자 눈높이 한 줄 요약",
        "mindset": "초보 투자자가 가져야 할 구체적인 대응 전략 (<ul><li> 사용)"
      }}
    }}
    """

    daily_briefing = {}
    try:
        # ⭐️ 깃허브 Secrets에서 불러온 API 키 1개를 사용
        if GEMINI_API_KEYS:
            genai.configure(api_key=GEMINI_API_KEYS[0]) 
            
            # ⭐️ 하루 1,500회 무료인 최신 1.5 버전 사용 (한도 문제 해결)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    
                    import re
                    json_match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
                    
                    if json_match:
                        clean_text = json_match.group(0)
                        daily_briefing = json.loads(clean_text, strict=False)
                        print("✅ 마켓 브리핑 생성 완료!")
                        break 
                    else:
                        raise ValueError("응답에서 JSON 형식을 찾을 수 없습니다.")
                        
                except Exception as inner_e:
                    print(f"   [!] {attempt+1}차 브리핑 파싱 실패: {inner_e}")
                    if attempt == max_retries - 1:
                        raise 
                    time.sleep(3) 
        else:
            print("⚠️ API 키가 설정되지 않아 브리핑 생성을 건너뜁니다.")
    except Exception as e:
        print(f"⚠️ 브리핑 생성 최종 실패: {e}")
        daily_briefing = {
            "hero_title": "⚠️ 시장 분석 데이터 지연",
            "hero_body": "API 데이터 파싱 문제로 인해 텍스트 분석 데이터를 불러오지 못했습니다.",
            "timeline": [{"date": "Error", "trend": "-", "title": "분석 실패", "desc": "데이터를 불러오는 중 문제가 발생했습니다."}],
            "morning_notes": [],
            "expert_views": [{"name": "시스템 알림", "desc": "뉴스 데이터 분석에 실패했습니다."}],
            "beginner_guide": {"summary": "현재 시황을 요약할 수 없습니다.", "mindset": "잠시 후 다시 실행해 주세요."}
        }

    report_path = generate_report(results_df, market_env, market_date, top20_summaries=daily_briefing, output_path=output)
    
    csv_path = report_path.replace(".html", ".csv")
    try: results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    except: pass
    
    # GitHub Pages 호스팅을 위해 index.html 복사
    import shutil
    try:
        shutil.copyfile(report_path, "index.html")
    except Exception as e:
        print(f"  ⚠️ index.html 복사 실패: {e}")
    
    print(f"\n✅ 완료! 리포트가 저장되었습니다: {report_path}")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("\n" + "!"*60)
        print("  🚨 프로그램 실행 중 에러가 발생했습니다!")
        print("!"*60 + "\n")
        traceback.print_exc()
