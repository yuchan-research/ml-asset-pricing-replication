
import pandas as pd
import numpy as np

def add_features(df,mode="baseline"):
    df = df.sort_values(["ticker", "date"]).copy()

    # =========================
    # 1. PRICE TREND 
    # =========================

    # Momentum
    df["mom_1m"] = df.groupby("ticker")["price"].pct_change(21)
    df["mom_3m"] = df.groupby("ticker")["price"].pct_change(63)
    df["mom_6m"] = df.groupby("ticker")["price"].pct_change(126)
    df["mom_12m"] = df.groupby("ticker")["price"].pct_change(252)

    # Momentum spread 
    df["mom_3m_12m"] = df["mom_3m"] - df["mom_12m"]

    # Short-term reversal
    df["rev_1w"] = df.groupby("ticker")["return"].shift(5)
    df["rev_1m"] = df.groupby("ticker")["return"].shift(21)

    #log
    df["log_price"] = np.log(df["price"])

    # =========================
    # 2. VOLATILITY / RISK
    # =========================

    df["vol_1m"] = df.groupby("ticker")["return"].rolling(21).std().reset_index(level=0, drop=True)
    df["vol_3m"] = df.groupby("ticker")["return"].rolling(63).std().reset_index(level=0, drop=True)
    df["vol_6m"] = df.groupby("ticker")["return"].rolling(126).std().reset_index(level=0, drop=True)

    # downside volatility
    df["downside_vol"] = df.groupby("ticker")["return"].apply(
        lambda x: x.rolling(21).apply(lambda r: r[r < 0].std() if len(r[r < 0]) > 0 else 0)
    ).reset_index(level=0, drop=True)

    # =========================
    # 3. LIQUIDITY (proxy)
    # =========================

    # price 기반 proxy (실제는 volume 필요)
    df["price_level"] = df["price"]

    # turnover proxy (근사)
    df["ret_abs"] = df["return"].abs()
    df["illiq_proxy"] = df["ret_abs"] / df["price"]

    #log
    df["log_illiq"] = np.log(df["illiq_proxy"] + 1e-8)

    # =========================
    # 4. TREND / MEAN REVERSION MIX
    # =========================

    # moving average
    df["ma_20"] = df.groupby("ticker")["price"].transform(lambda x: x.rolling(20).mean())
    df["ma_60"] = df.groupby("ticker")["price"].transform(lambda x: x.rolling(60).mean())

    df["ma_ratio_20"] = df["price"] / df["ma_20"]
    df["ma_ratio_60"] = df["price"] / df["ma_60"]

    # =========================
    # 5. RETURN LAGS 
    # =========================

    df["ret_lag_1"] = df.groupby("ticker")["return"].shift(1)
    df["ret_lag_2"] = df.groupby("ticker")["return"].shift(2)
    df["ret_lag_5"] = df.groupby("ticker")["return"].shift(5)
    df["ret_lag_10"] = df.groupby("ticker")["return"].shift(10)
    df["ret_lag_21"] = df.groupby("ticker")["return"].shift(21)


    if mode == "improved":
    # =========================
    # 6. interaction feature (데이터 수가 적음 -> 직접 만듦(다른 카테고리끼리))
    # =========================

    # Momentum × Volatility
      df["mom_vol_1"] = df["mom_3m"] * df["vol_1m"]
      df["mom_vol_2"] = df["mom_6m"] * df["vol_3m"]

    # Momentum × Reversal
      df["mom_rev"] = df["mom_3m"] * df["rev_1m"]

    # Volatility × Downside
      df["vol_down"] = df["vol_1m"] * df["downside_vol"]

    # Reversal × Volatility
      df["rev_vol"] = df["rev_1m"] * df["vol_1m"]

    # Liquidity proxy × Momentum
      df["illiq_mom"] = df["illiq_proxy"] * df["mom_3m"]

      df["mom_logvol"] = df["mom_3m"] * np.log(df["vol_1m"] + 1e-8)

    #특정 조건에서만 작동하는 signal(regime-dependent signal)
      df["high_vol_flag"] = (df["vol_1m"] > df["vol_1m"].median()).astype(int)
      df["mom_high_vol"] = df["mom_3m"] * df["high_vol_flag"]

    return df
