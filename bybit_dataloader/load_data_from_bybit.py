from functions import get_klines_bybit
from config import *
import pandas as pd
import os

if __name__ == "__main__":
    all_data = []

    current_dir = os.path.dirname(os.path.abspath(__file__))

    klines_dir = os.path.join(os.path.dirname(current_dir), 'klines_data')

    os.makedirs(klines_dir, exist_ok=True)

    for symbol in symbols:
        print(f"start dataload for {symbol}")
        df = get_klines_bybit(symbol, start_date, end_date, interval)
        all_data.append(df)
        print(f"end dataload for {symbol}, {len(df)=}")

    dataset = pd.concat(all_data)
    print(f"dataset before drop duplicate {len(dataset)=}")
    dataset = dataset.drop_duplicates(["symbol", "timestamp"])
    print(f"dataset after drop duplicate {len(dataset)=}")

    dataset = dataset.sort_values(["symbol", "timestamp"])

    output_path = os.path.join(klines_dir, f"crypto_{interval}m_bybit_TEST.parquet")
    dataset.to_parquet(output_path, engine="pyarrow")

    print("Saved rows:", len(dataset))
    print(f"File saved to: {output_path}")