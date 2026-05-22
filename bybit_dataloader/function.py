from config import *
import time
import pandas as pd

def get_klines_bybit(symbol, start_date, end_date, interval, limit=1000):

    rows = []

    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)

    while True:

        response = session.get_kline(
            category="spot",
            symbol=symbol,
            interval=interval,
            end=end_ts,
            limit=limit,
        )

        data = response["result"]["list"]

        if not data:
            break

        rows.extend(data)

        oldest_ts = int(data[-1][0])

        if oldest_ts <= start_ts:
            break

        end_ts = oldest_ts - 1

        time.sleep(0.5)

    df = pd.DataFrame(rows, columns=[
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover"
    ])

    df["symbol"] = symbol

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)

    df = df[df["timestamp"] >= start_date]

    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })

    return df