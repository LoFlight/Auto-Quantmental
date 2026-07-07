"""
================================================================
  퀀트멘털 AI 학습 시스템 v4.0 — 통합 최종본
================================================================
  [v2_0 대비 수정 사항 6가지]
  Fix1  최종 모델 scaler를 train 80%로만 fit → 누수 완전 제거
  Fix2  CV best_iteration 평균 → 최종 모델 n_estimators 적용
  Fix3  볼린저밴드·MACD, 데이터 부족 시 자동 비활성화
  Fix4  증분 학습 (init_model) 추가 → 매일 수초 완료
  Fix5  시장 매크로 피처 → 종목별 상대 지표로 교체 (변별력 향상)
  Fix6  학습 후 MAE 모니터링 + 드리프트 경고 자동화

  [실행 방법]
  python ai_trainer_final.py                      ← 대화형 실행
  python ai_trainer_final.py --mode status        ← 현황 확인
  python ai_trainer_final.py --mode train         ← 완전 재학습
  python ai_trainer_final.py --mode update        ← 증분 학습 (매일)
  python ai_trainer_final.py --ticker NVDA        ← 종목별 예측
================================================================
"""

import sys, os, warnings, pickle, json, time, argparse
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 패키지 자동 설치 ────────────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for mod, pkg in [("pandas","pandas"), ("numpy","numpy"),
                 ("sklearn","scikit-learn"), ("lightgbm","lightgbm"),
                 ("tqdm","tqdm")]:
    try: __import__(mod)
    except ImportError:
        print(f"  📦 {pkg} 설치 중..."); _pip(pkg)

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


# ════════════════════════════════════════════════════════════
#  설정
# ════════════════════════════════════════════════════════════
MODEL_DIR = Path("quantamental_models")
MODEL_DIR.mkdir(exist_ok=True)
STATE_FILE = MODEL_DIR / "state.json"

HORIZONS = {
    "T1":1, "T5":5, "T21":21, "T63":63, "T126":126, "T252":252
}
HORIZON_LABELS = {
    "T1":"다음날", "T5":"1주 후", "T21":"1개월 후",
    "T63":"3개월 후", "T126":"6개월 후", "T252":"1년 후",
}
# 각 기간별 최소 거래일
MIN_DAYS = {
    "T1":60, "T5":60, "T21":80, "T63":100, "T126":180, "T252":320
}
# 매일 증분 추가 트리 수
INCR_TREES = 30

RAW_FEATURES = [
    "p1","p2","p3","p4","p5","p6","total",
    "rsi","vol_ratio","pct_change",
]
DERIVED_FEATURES = [
    # 기본 파생
    "price_to_ma120",   # 추세 강도
    "p1_p2_ratio",      # 펀더멘털/밸류 균형
    "total_rank_pct",   # 당일 전체 순위 백분위
    "rsi_zone",         # RSI 구간
    "vol_spike",        # 거래량 급증
    "p5_p3_delta",      # 기술적-심리 괴리
    # 기술적 지표 (Fix3: 데이터 충분할 때만 유효)
    "bb_width",         # 볼린저밴드 폭 (변동성)
    "bb_position",      # 볼린저밴드 내 위치
    "macd_hist",        # MACD 히스토그램
    # 상대 시장 지표 (Fix5: 단순 평균 → 상대값)
    "rel_momentum",     # 자신 pct - 시장 평균 pct (상대 모멘텀)
    "rel_rsi",          # 자신 RSI - 시장 평균 RSI (상대 RSI)
    "market_breadth",   # 상승 종목 비율 (시장 환경)
]
ALL_FEATURES = RAW_FEATURES + DERIVED_FEATURES


# ════════════════════════════════════════════════════════════
#  1. 데이터 전처리
# ════════════════════════════════════════════════════════════

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"]).reset_index(drop=True)
    df["zero_flag"] = df["zero_flag"].astype(str).str.lower().eq("true").astype(float)

    n_days = df["date"].nunique()

    # ── 기본 파생 피처 ────────────────────────────────────
    ma120 = df.get("ma120", pd.Series(np.nan, index=df.index))
    df["price_to_ma120"] = (df["price"] / ma120.where(ma120>0)).fillna(1.0)
    df["p1_p2_ratio"]    = (df["p1"] / df["p2"].replace(0, np.nan)).fillna(1.0)
    df["total_rank_pct"] = df.groupby("date")["total"].rank(pct=True)
    df["rsi_zone"]       = pd.cut(df["rsi"], bins=[-np.inf,30,70,np.inf],
                                  labels=[0,1,2]).astype(float).fillna(1.0)
    df["vol_spike"]      = (df["vol_ratio"] > 1.5).astype(float)
    df["p5_p3_delta"]    = df["p5"] - df["p3"]

    # ── 기술적 지표 (Fix3: 충분한 데이터 있을 때만) ──────
    # pandas 3.x 호환: groupby().apply() 대신 for 루프 사용
    tech_chunks = []
    for tkr, grp in df.groupby("ticker"):
        grp = grp.copy()

        # 볼린저밴드: 최소 20거래일 필요
        if len(grp) >= 20:
            ma20  = grp["price"].rolling(20, min_periods=20).mean()
            std20 = grp["price"].rolling(20, min_periods=20).std()
            bb_up = ma20 + std20 * 2
            bb_lo = ma20 - std20 * 2
            grp["bb_width"]    = ((bb_up - bb_lo) / ma20.replace(0, np.nan)).fillna(0)
            grp["bb_position"] = ((grp["price"] - bb_lo) / (bb_up - bb_lo + 1e-8)).clip(0, 1).fillna(0.5)
        else:
            grp["bb_width"]    = 0.0   # 데이터 부족 → 중립
            grp["bb_position"] = 0.5

        # MACD: 최소 35거래일 필요
        if len(grp) >= 35:
            ema12 = grp["price"].ewm(span=12, adjust=False).mean()
            ema26 = grp["price"].ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9, adjust=False).mean()
            grp["macd_hist"] = (macd - sig).fillna(0)
        else:
            grp["macd_hist"] = 0.0     # 데이터 부족 → 중립

        tech_chunks.append(grp)

    df = pd.concat(tech_chunks).sort_values(["ticker","date"]).reset_index(drop=True)

    # ── 상대 시장 지표 (Fix5) ──────────────────────────────
    mkt = df.groupby("date").agg(
        mkt_pct=("pct_change","mean"),
        mkt_rsi=("rsi","mean"),
        market_breadth=("pct_change", lambda x: (x>0).mean())
    ).reset_index()
    df = df.merge(mkt, on="date", how="left")
    df["rel_momentum"] = df["pct_change"] - df["mkt_pct"]   # 시장 대비 초과 등락률
    df["rel_rsi"]      = df["rsi"]        - df["mkt_rsi"]   # 시장 대비 상대 RSI
    df.drop(columns=["mkt_pct","mkt_rsi"], inplace=True)

    # ── 타겟 라벨 ─────────────────────────────────────────
    for h, days in HORIZONS.items():
        df[f"target_{h}"] = (
            df.groupby("ticker")["price"]
            .shift(-days)
            .div(df["price"])
            .sub(1).mul(100)
        )

    # 누락 컬럼 보완
    for f in ALL_FEATURES:
        if f not in df.columns:
            df[f] = 0.0

    df.drop(columns=["ma120","name"], errors="ignore", inplace=True)
    return df


# ════════════════════════════════════════════════════════════
#  2. 상태 관리
# ════════════════════════════════════════════════════════════

class State:
    def __init__(self):
        self.d = json.loads(STATE_FILE.read_text(encoding="utf-8")) \
                 if STATE_FILE.exists() else \
                 {"trained_dates":[], "horizons":{}, "last_full":{}}

    def save(self):
        STATE_FILE.write_text(json.dumps(self.d, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    def new_dates(self, df):
        seen = set(self.d["trained_dates"])
        all_ = set(df["date"].dt.strftime("%Y-%m-%d").unique())
        return sorted(all_ - seen)

    def mark(self, dates):
        self.d["trained_dates"] = sorted(set(self.d["trained_dates"]) | set(dates))
        self.save()

    def hset(self, h, meta):
        self.d["horizons"][h] = meta; self.save()

    def hget(self, h):
        return self.d["horizons"].get(h, {})

    def need_full(self, h):
        m = self.hget(h)
        if not m: return True, "최초 학습"
        days = (datetime.now() - datetime.fromisoformat(m["trained_at"])).days
        if m.get("is_initial") and days >= 30:
            return True, "초기 모델 정교화 (30일 경과)"
        cv_mae  = m.get("cv_mae", 99)
        rec_mae = m.get("recent_mae", cv_mae)
        if rec_mae > cv_mae * 1.3 and rec_mae > 2.0:
            return True, f"성능 악화 ({cv_mae:.2f}→{rec_mae:.2f}%)"
        if days >= 90:
            return True, f"정기 재학습 ({days}일 경과)"
        return False, f"증분 유지 ({days}일 전)"


# ════════════════════════════════════════════════════════════
#  3. 학습 엔진
# ════════════════════════════════════════════════════════════

LGB_BASE = dict(
    objective="regression", metric="mae",
    n_estimators=500, learning_rate=0.03,
    num_leaves=31, max_depth=5,
    min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1,
    verbose=-1, n_jobs=-1, random_state=42,
)


def full_train(df, state, targets=None, force=False, silent=False):
    """
    완전 재학습 (Fix1 + Fix2 적용)
    Fix1: train 80% 로만 scaler.fit
    Fix2: CV best_iteration 평균 → 최종 n_estimators
    """
    targets = targets or list(HORIZONS.keys())
    n_days  = df["date"].nunique()
    boosters, scalers = {}, {}

    if MODEL_DIR.joinpath("boosters.pkl").exists():
        boosters, scalers = _load_raw()

    bar = tqdm(targets, desc="🤖 학습 진행", disable=silent,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for h in bar:
        bar.set_postfix_str(HORIZON_LABELS[h])
        target_col = f"target_{h}"
        valid = df[df[target_col].notna()].copy()

        if len(valid) < MIN_DAYS[h] * 5:
            if not silent:
                print(f"  ⏭  {HORIZON_LABELS[h]}: 데이터 부족 ({n_days}/{MIN_DAYS[h]}일)")
            continue

        need, reason = state.need_full(h)
        if not force and not need and h in boosters:
            if not silent:
                print(f"  ⏭  {HORIZON_LABELS[h]}: 재학습 불필요 ({reason})")
            continue

        X = valid[ALL_FEATURES].fillna(0)
        y = valid[target_col].clip(-50, 50)

        # Fix2: CV에서 best_iteration 수집
        tscv = TimeSeriesSplit(n_splits=5)
        best_iters, cv_maes = [], []
        n_train = int(len(X) * 0.8)

        for tr_idx, val_idx in tscv.split(X):
            # Fix1: train 구간으로만 scaler fit
            fold_sc = RobustScaler().fit(X.iloc[tr_idx])
            Xtr = fold_sc.transform(X.iloc[tr_idx])
            Xvl = fold_sc.transform(X.iloc[val_idx])
            ytr, yvl = y.iloc[tr_idx], y.iloc[val_idx]

            m = lgb.LGBMRegressor(**LGB_BASE)
            m.fit(Xtr, ytr,
                  eval_set=[(Xvl, yvl)],
                  callbacks=[lgb.early_stopping(40, verbose=False),
                              lgb.log_evaluation(-1)])
            best_iters.append(m.best_iteration_)
            cv_maes.append(mean_absolute_error(yvl, m.predict(Xvl)))

        # Fix1: 최종 모델도 train 80%로만 scaler fit
        final_sc = RobustScaler().fit(X.iloc[:n_train])
        Xf = final_sc.transform(X)

        # Fix2: best_iteration 평균으로 최종 트리 수 결정
        best_n = int(np.mean(best_iters))
        params = dict(LGB_BASE, n_estimators=best_n)
        final  = lgb.LGBMRegressor(**params)
        final.fit(Xf, y, callbacks=[lgb.log_evaluation(-1)])

        boosters[h] = final.booster_
        scalers[h]  = final_sc

        mae_cv = float(np.mean(cv_maes))
        top3 = sorted(zip(ALL_FEATURES, final.feature_importances_),
                      key=lambda x:-x[1])[:3]
        state.hset(h, {
            "trained_at":  datetime.now().isoformat(),
            "n_samples":   len(valid),
            "cv_mae":      round(mae_cv, 4),
            "recent_mae":  round(mae_cv, 4),
            "best_iter":   best_n,
            "is_initial":  not force,
            "label":       HORIZON_LABELS[h],
            "top3":        [f[0] for f in top3],
        })

        if not silent:
            print(f"  ✅ {HORIZON_LABELS[h]:8s}  MAE={mae_cv:.2f}%  "
                  f"트리={best_n}  중요피처={[f[0] for f in top3]}")

    _save_raw(boosters, scalers)
    all_dates = df["date"].dt.strftime("%Y-%m-%d").unique().tolist()
    state.mark(all_dates)
    state.d["last_full"][",".join(targets)] = datetime.now().isoformat()
    state.save()
    return boosters, scalers


def incremental_update(df, state):
    """증분 학습 — 새 날짜 데이터만 수초 안에 처리 (Fix4)"""
    new_dates = state.new_dates(df)
    if not new_dates:
        print("  ✅ 새 데이터 없음 — 업데이트 불필요")
        return

    if not MODEL_DIR.joinpath("boosters.pkl").exists():
        print("  ⚠️  저장 모델 없음 → 완전 학습으로 전환")
        full_train(df, state)
        return

    print(f"  ⚡ 증분 학습: {new_dates}")
    boosters, scalers = _load_raw()
    t0 = time.time()

    for h in HORIZONS:
        if h not in boosters: continue
        target_col = f"target_{h}"

        need, reason = state.need_full(h)
        if need:
            print(f"\n  ⚠️  {HORIZON_LABELS[h]}: {reason} → 완전 재학습")
            full_train(df, state, targets=[h], force=True, silent=True)
            boosters, scalers = _load_raw()
            continue

        new_df = df[df["date"].dt.strftime("%Y-%m-%d").isin(new_dates)]
        valid  = new_df[new_df[target_col].notna()]
        if valid.empty:
            continue

        X  = valid[ALL_FEATURES].fillna(0)
        y  = valid[target_col].clip(-50, 50)
        Xs = scalers[h].transform(X)

        incr = lgb.LGBMRegressor(**(dict(LGB_BASE, n_estimators=INCR_TREES)))
        incr.fit(Xs, y, init_model=boosters[h], callbacks=[lgb.log_evaluation(-1)])
        boosters[h] = incr.booster_

        # Fix6: 최근 30일 MAE 모니터링
        recent_val = df[df["date"] >= df["date"].max() - pd.Timedelta(days=30)]
        recent_val = recent_val[recent_val[target_col].notna()]
        if not recent_val.empty:
            Xv = scalers[h].transform(recent_val[ALL_FEATURES].fillna(0))
            yv = recent_val[target_col].clip(-50, 50)
            recent_mae = float(mean_absolute_error(yv, incr.predict(Xv)))
            m = state.hget(h)
            prev_mae = m.get("cv_mae", 99)
            m["recent_mae"]  = round(recent_mae, 4)
            m["last_update"] = datetime.now().isoformat()
            m["total_trees"] = incr.booster_.num_trees()
            state.hset(h, m)
            drift = recent_mae > prev_mae * 1.3
            flag  = "⚠️  성능 악화!" if drift else "✅"
            print(f"  {flag} {HORIZON_LABELS[h]:8s}  +{INCR_TREES}트리  "
                  f"최근MAE={recent_mae:.2f}%  (기준={prev_mae:.2f}%)")
        else:
            n_tr = incr.booster_.num_trees()
            print(f"  ✅ {HORIZON_LABELS[h]:8s}  +{INCR_TREES}트리 (총 {n_tr})")

    _save_raw(boosters, scalers)
    state.mark(new_dates)
    print(f"\n  ⏱  총 소요: {time.time()-t0:.1f}초")


# ════════════════════════════════════════════════════════════
#  4. 저장/불러오기
# ════════════════════════════════════════════════════════════

def _save_raw(boosters, scalers):
    with open(MODEL_DIR/"boosters.pkl","wb") as f: pickle.dump(boosters, f)
    with open(MODEL_DIR/"scalers.pkl","wb")  as f: pickle.dump(scalers, f)

def _load_raw():
    with open(MODEL_DIR/"boosters.pkl","rb") as f: b = pickle.load(f)
    with open(MODEL_DIR/"scalers.pkl","rb")  as f: s = pickle.load(f)
    return b, s

def models_exist():
    return (MODEL_DIR/"boosters.pkl").exists()


# ════════════════════════════════════════════════════════════
#  5. 리포트 출력
# ════════════════════════════════════════════════════════════

def full_report(df, sort_by="T21"):
    """전체 종목 통합 예측 리포트"""
    if not models_exist():
        print("  ❌ 학습된 모델이 없습니다."); return

    boosters, scalers = _load_raw()
    state = State()

    latest_date = df["date"].max()
    today_df    = df[df["date"] == latest_date].copy()

    results = []
    for _, row in today_df.iterrows():
        X_row = row.reindex(ALL_FEATURES).fillna(0).values.reshape(1,-1)
        preds = {}
        for h in HORIZONS:
            if h in boosters:
                Xs = scalers[h].transform(X_row)
                preds[h] = float(boosters[h].predict(Xs)[0])
        results.append({"ticker": row["ticker"],
                         "price":  row["price"],
                         "total":  row.get("total", 0),
                         **preds})

    res = pd.DataFrame(results)
    if sort_by in res.columns:
        res = res.sort_values(sort_by, ascending=False)
    report_path = "ai_prediction_report.csv"
    res.to_csv(report_path, index=False, encoding="utf-8-sig")
    
# 모델 MAE 헤더
    maes = {h: state.hget(h).get("recent_mae") or state.hget(h).get("cv_mae")
            for h in HORIZONS if state.hget(h)}

    print(f"\n{'='*100}")
    print(f"  📊 나스닥 100 AI 예측 리포트  기준일: {latest_date.strftime('%Y-%m-%d')}")
    print(f"  정렬: {HORIZON_LABELS.get(sort_by, sort_by)} 예측 수익률  "
          f"| 모델 MAE: " + " / ".join(
              f"{HORIZON_LABELS[h]}={v:.1f}%" for h,v in maes.items() if v))
    print(f"{'='*100}")
    print(f"  {'티커':6s} | {'현재가':>8s} | {'총점':>5s} | "
          f"{'다음날':>8s} | {'1주 후':>8s} | {'1개월':>8s} | "
          f"{'3개월':>8s} | {'6개월':>8s} | {'1년 후':>8s}")
    print("  " + "─"*96)

    for _, r in res.iterrows():
        def fmt(v): return f"{v:>+7.1f}%" if pd.notna(v) else "    N/A "
        emoji = ("🚀" if r.get("T21",0)>=15 else
                 "📈" if r.get("T21",0)>=5  else
                 "➡️ " if r.get("T21",0)>=-5 else "📉")
        print(f"  {r['ticker']:6s} | ${r['price']:>7.2f} | {r['total']:>4.0f}점 | "
              f"{fmt(r.get('T1'))} | {fmt(r.get('T5'))} | {fmt(r.get('T21'))} | "
              f"{fmt(r.get('T63'))} | {fmt(r.get('T126'))} | {fmt(r.get('T252'))}  {emoji}")

    print(f"  {'─'*96}")
    print(f"  * 주의: MAE 범위 안에서는 방향성만 참고하세요. 투자 손익의 책임은 투자자 본인에게 있습니다.\n")


def ticker_report(df, ticker):
    """단일 종목 상세 예측"""
    if not models_exist():
        print("  ❌ 학습된 모델이 없습니다."); return

    boosters, scalers = _load_raw()
    state = State()

    tdf = df[df["ticker"] == ticker.upper()].sort_values("date")
    if tdf.empty:
        sim = [t for t in df["ticker"].unique() if ticker[:3].upper() in t]
        print(f"  ❌ '{ticker}' 없음. 유사: {sim[:5]}"); return

    row  = tdf.iloc[-1]
    cur  = row["price"]
    tot  = row.get("total", 0)
    date = row["date"].strftime("%Y-%m-%d")

    grade = next(g for t,g in [(95,"🏆 APEX"),(83,"🟢🟢 강력매수"),
                               (72,"🟢 매수"),(60,"🔵 보유"),
                               (50,"⚪ 주의"),(-999,"🔴 매도")] if tot>=t)

    print(f"\n{'='*60}")
    print(f"  🔮 {ticker.upper()} 예측 리포트")
    print(f"{'='*60}")
    print(f"  기준일: {date}  현재가: ${cur:.2f}  총점: {tot:.1f}/110  {grade}\n")

    X_row = row.reindex(ALL_FEATURES).fillna(0).values.reshape(1,-1)
    print(f"  {'기간':10s}  {'예측수익률':>11s}  {'예측가':>10s}  {'신뢰도':>10s}  판단")
    print("  " + "─"*56)

    for h, label in HORIZON_LABELS.items():
        if h not in boosters:
            print(f"  {label:10s}  {'(학습 대기)':>32s}"); continue

        Xs   = scalers[h].transform(X_row)
        pred = float(boosters[h].predict(Xs)[0])
        pp   = cur * (1 + pred/100)
        m    = state.hget(h)
        mae  = m.get("recent_mae") or m.get("cv_mae") or 99
        conf = "★★★" if mae<3 else "★★☆" if mae<7 else "★☆☆"
        judge = ("🚀 강한상승" if pred>=20 else "📈 상승" if pred>=10 else
                 "↗️  소폭상승" if pred>=3 else "➡️  횡보" if pred>=-3 else
                 "↘️  소폭하락" if pred>=-10 else "📉 하락")
        sign = "+" if pred>=0 else ""
        print(f"  {label:10s}  {sign}{pred:>7.1f}%   ${pp:>9.2f}  {conf:>10s}  {judge}")

    print()


def show_status(df):
    """데이터 현황 + 모델 상태"""
    state  = State()
    dates  = sorted(df["date"].unique())
    n_days = len(dates)
    new    = state.new_dates(df)

    print(f"\n{'='*62}")
    print(f"  📊 퀀트멘털 AI 시스템 현황 v4.0")
    print(f"{'='*62}")
    print(f"  기간: {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')}")
    print(f"  거래일: {n_days}일  /  종목: {df['ticker'].nunique()}개  /  전체 행: {len(df):,}")
    if new: print(f"  ⚡ 미학습 날짜: {new}")
    print()
    print(f"  {'기간':12s}  {'필요':>6s}  {'현재':>6s}  {'MAE':>7s}  {'트리':>5s}  상태")
    print("  " + "─"*54)
    for h, label in HORIZON_LABELS.items():
        m    = state.hget(h)
        mae  = f"{m.get('recent_mae') or m.get('cv_mae', 0):.2f}%" if m else "─"
        trees = str(m.get("total_trees", m.get("best_iter","─"))) if m else "─"
        bar  = "✅ 가능" if n_days>=MIN_DAYS[h] else f"⏳ {MIN_DAYS[h]-n_days}일 더"
        print(f"  {label:12s}  {MIN_DAYS[h]:>5d}일  {n_days:>5d}일  {mae:>7s}  {trees:>5s}개  {bar}")

    print(f"\n  [명령어 가이드]")
    s = "ai_trainer_final.py"
    print(f"  python {s} --mode update            ← 매일 실행 (증분 학습)")
    print(f"  python {s} --mode train             ← 최초/완전 재학습")
    print(f"  python {s} --ticker NVDA            ← 종목별 예측")
    print(f"  python {s} --mode report            ← 전체 순위 리포트")
    print(f"  python {s} --mode report --sort T63 ← 3개월 기준 정렬")


# ════════════════════════════════════════════════════════════
#  6. 메인
# ════════════════════════════════════════════════════════════

def interactive(df, state):
    """대화형 실행"""
    print(f"\n{'='*60}")
    print(f"  🤖 퀀트멘털 AI v4.0 — 대화형 모드")
    print(f"{'='*60}")

    n_days = df["date"].nunique()
    months = n_days / 21
    print(f"\n[1] 데이터: {months:.1f}개월 ({n_days} 거래일)")

    new = state.new_dates(df)
    if new and models_exist():
        print(f"\n[!] 새 데이터 {len(new)}일 감지 → 증분 학습 권장")
        ans = input("    증분 학습을 진행할까요? (Y/n): ").strip().lower()
        if ans != "n":
            incremental_update(df, state)
    else:
        ans = input("\n[2] AI 학습을 새로 진행할까요? (Y/n): ").strip().lower()
        if ans != "n":
            full_train(df, state)

    print("\n[3] 예측 리포트 생성 중...")
    full_report(df)


def main():
    parser = argparse.ArgumentParser(description="퀀트멘털 AI v4.0")
    parser.add_argument("--data",   default="data/nasdaq100data.csv")
    parser.add_argument("--mode",   choices=["status","train","update","report"],
                        default=None)
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--sort",   default="T21", choices=list(HORIZONS))
    parser.add_argument("--force",  action="store_true")
    args = parser.parse_args()

    csv = args.data
    if not os.path.exists(csv):
        print(f"❌ {csv} 없음"); sys.exit(1)

    print("\n[데이터 로드 중...]", end=" ", flush=True)
    df    = load_and_preprocess(csv)
    state = State()
    print("완료")

    if args.ticker:
        ticker_report(df, args.ticker)
    elif args.mode == "status":
        show_status(df)
    elif args.mode == "train":
        full_train(df, state, force=args.force)
        print("\n다음 실행 시 증분 학습이 자동 적용됩니다.")
    elif args.mode == "update":
        incremental_update(df, state)
    elif args.mode == "report":
        full_report(df, sort_by=args.sort)
    else:
        interactive(df, state)


if __name__ == "__main__":
    main()
