import numpy as np
import pandas as pd

# Out-of-sample R^2 (모델이 평균 예측보다 얼마나 잘 맞추는지)
# (금융에서는 R^2 주로 낮게 나옴 => 다른 지표 같이 봐야함)
def oos_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = ((y_true - y_true.mean()) ** 2).mean()
    if denom == 0:
        return np.nan
    return 1 - ((y_true - y_pred) ** 2).mean() / denom

# 상위 10% = long / 하위 10% = short 포트폴리오 수익률
def long_short_portfolio(df, pred_col="pred", target_col="target", long_q=0.9, short_q=0.1):
    df = df.copy()
    df = df.dropna(subset=[pred_col, target_col])

    df["rank"] = df.groupby("date")[pred_col].rank(pct=True)

    long = df[df["rank"] >= long_q]
    short = df[df["rank"] <= short_q]

    long_ret = long.groupby("date")[target_col].mean()
    short_ret = short.groupby("date")[target_col].mean()

    ls = (long_ret - short_ret).dropna()
    ls.name = "long_short_return"
    return ls

# Sharpe Ratio (위험 대비 수익)( (수익률-무위험수익률)/변동성 )
def sharpe(returns):
    returns = pd.Series(returns).dropna()
    if returns.std() == 0:
        return np.nan
    return returns.mean() / returns.std() * (252 ** 0.5)
