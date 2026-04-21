import pandas as pd
from src.data_loader import load_data
from src.data_loader import load_data_fixed
from src.data_loader import sample_universe
from src.feature_engineering import add_features
from src.dataset import create_target, normalize_features, split_data, get_feature_cols
from src.models.linear import train_ols
from src.models.elastic_net import train_enet
from src.models.tree import train_rf
from src.evaluate import oos_r2, long_short_portfolio, sharpe

def prepare_data(mode="baseline"):
    if mode == "baseline":
      tickers = [
      'AAPL','MSFT','GOOG','AMZN','META','NVDA','TSLA','JPM','V','PG',
      'ADBE','CRM','ORCL','INTC','AMD',
      'JNJ','PFE','MRK','ABBV','TMO',
      'KO','PEP','COST','WMT','HD',
      'XOM','CVX','BA','CAT','GE',
      'NFLX','DIS','MA','PYPL','BRK-B','GS','UPS','UNP','LIN','AMAT']
      df = load_data_fixed(tickers)
    else:
      df = load_data()
    print("1 clear")

    df = add_features(df,mode)
    print("2 clear")

    df = create_target(df)
    print("3 clear")

    if mode == "improved":
      df = sample_universe(df, n=100, method="random")
      print("sampling clear")

    df = df.dropna(subset=["target"]).reset_index(drop=True)

    feature_cols = get_feature_cols(df)

    df = normalize_features(df, feature_cols)

    train, val, test = split_data(df)

    train = train.dropna(subset=feature_cols + ["target"]).copy()
    val = val.dropna(subset=feature_cols + ["target"]).copy()
    test = test.dropna(subset=feature_cols + ["target"]).copy()

    return train, val, test, feature_cols

def run_model(train, val, test, feature_cols, model_name="rf"):

    X_train = train[feature_cols].values
    y_train = train["target"].values

    X_test = test[feature_cols].values
    y_test = test["target"].values

    if model_name == "ols":
        model = train_ols(X_train, y_train)
    elif model_name == "enet":
        model = train_enet(X_train, y_train)
    elif model_name == "rf":
        model = train_rf(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    test = test.copy()
    test["pred"] = model.predict(X_test)

    r2 = oos_r2(y_test, test["pred"].values)
    ls = long_short_portfolio(test, pred_col="pred", target_col="target")
    sh = sharpe(ls)

    return {
        "model": model_name,
        "oos_r2": r2,
        "sharpe": sh,
    }
