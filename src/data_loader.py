import yfinance as yf
import pandas as pd
import requests


def get_sp500_tickers(path="/content/drive/MyDrive/ml_asset_pricing/data/sp500_tickers_2026.csv"):

    df = pd.read_csv(path)

    # 컬럼 이름이 없으면 첫 컬럼 사용
    tickers = df.iloc[:, 0].tolist()

    # yfinance 호환 (혹시 필요하면)
    tickers = [t.replace(".", "-") for t in tickers]

    return tickers

def download_data(tickers, start="2010-01-01", end="2024-12-31"):

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True, # Close가 Adj Close(조정 종가): 배당 + 액면분할까지 반영된 가격 => 수익률 계산 시 사용
        progress=False
    )

    # 조용한 실패를 막는 장치
    if raw.empty:
        raise ValueError("No data returned from yfinance.")

    # 여러 티커면 MultiIndex, 한 티커면 일반 DataFrame
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        if isinstance(tickers, str):
            close.columns = [tickers]

    panel = close.stack().reset_index()
    panel.columns = ["date", "ticker", "price"]

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel["return"] = panel.groupby("ticker")["price"].pct_change()

    panel = panel.replace([float("inf"), float("-inf")], pd.NA)
    panel = panel.dropna(subset=["price", "return"]).reset_index(drop=True)

    # 티커별 데이터 개수 체크 & 데이터 정상 확인
   # counts = panel.groupby("ticker").size()
   # print(counts.describe())
   # print(panel["return"].describe())
   # print("티커별 데이터 정상인지 확인")

    return panel


def load_data():

    tickers = get_sp500_tickers()

    df = download_data(tickers)

    return df


def sample_universe(df, n=100, method="random"):

    def _sample(group):
        group = group.dropna()

        if len(group) < n:
            return group  # 부족하면 그냥 사용

        if method == "random":
            return group.sample(n=n, random_state=42)

        elif method == "topcap":
            return group.sort_values("market_cap", ascending=False).head(n)

        else:
            raise ValueError("method must be random or topcap")

    df = df.groupby("date", group_keys=False).apply(_sample)
    return df

def load_data_fixed(tickers, start="2010-01-01", end="2024-12-31"):
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True, 
        progress=False
    )
    
    if raw.empty:
        raise ValueError("No data returned from yfinance.")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        if isinstance(tickers, str):
            close.columns = [tickers]

    panel = close.stack().reset_index()
    panel.columns = ["date", "ticker", "price"]

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    panel["return"] = panel.groupby("ticker")["price"].pct_change()

    panel = panel.replace([float("inf"), float("-inf")], pd.NA)
    panel = panel.dropna(subset=["price", "return"]).reset_index(drop=True)

    return panel
