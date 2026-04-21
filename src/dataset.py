import pandas as pd

# target 데이터
def create_target(df):
    df = df.sort_values(["ticker", "date"]).copy()
    df["target"] = df.groupby("ticker")["return"].shift(-1)
    return df

# 같은 날짜 안에서 종목끼리 비교, 각 feature를 순위 기반으로 정규화
def normalize_features(df, feature_cols):
    df = df.copy()
    # cross-sectional rank normalization by date
    df[feature_cols] = df.groupby("date")[feature_cols].rank(pct=True)
    return df

# train(학습용)/val(튜닝용)/test(평가) 데이터 셋 나누기
def split_data(df, train_end="2015-12-31", val_end="2017-12-31"):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)

    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()

    return train, val, test

def get_feature_cols(df):
    exclude = {"date", "ticker", "price", "return", "target"}
    return [c for c in df.columns if c not in exclude]
