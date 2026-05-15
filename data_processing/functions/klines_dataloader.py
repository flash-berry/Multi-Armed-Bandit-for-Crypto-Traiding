import pandas as pd
from pathlib import Path

class KlinesDataLoader:
    """Загрузка данных OHLCV для symbols"""

    def __init__(self, symbols=None):
        if symbols is None:
            raise ValueError("Передайте название активов")

        if isinstance(symbols, str):
            symbols_list = [symbols]
        else:
            symbols_list = symbols

        self.symbols = symbols_list

    def statistics_df(self, dataframe):
        """Вывод статистики по датафрейму."""
        if dataframe is None or dataframe.empty:
            raise ValueError("Передан пустой DataFrame")

        print(f"Количество строк: {dataframe.shape[0]}\nКоличество столбцов: {dataframe.shape[1]}")
        print(f"Количество пропущенных значений: {dataframe.isna().sum().sum()}\n")
        print(f"Название столбцов: {', '.join(dataframe.columns.to_list())}")
        print(f"Название активов: {', '.join(self.symbols)}\n")
        for sym in self.symbols:
            print(f"Длина временного ряда актива {sym}: {dataframe[dataframe['symbol'] == sym].shape[0]}")
        print(f"\nВременные рамки ряда каждого актива:")
        for sym, ts in dataframe.groupby('symbol')['timestamp'].agg(['min', 'max']).iterrows():
            print(f"{sym}: {ts['min']} - {ts['max']}")

    def load_data(self, download_path, analyse_data=False, cleaning=False):
        '''
        Загрузка данных OHLCV по пути download_path.

        :param download_path: (str) путь к данным
        :param analyse_data: (bool) вывод статистики
        :param cleaning: (bool) выравнивание временных рядов и удаление 'turnover'
        :return: df: (DataFrame) датафрейм OHLCV
        '''
        if not isinstance(download_path, str) or not download_path.strip():
            raise ValueError(
                "download_path должен быть непустой строкой"
            )

        current_file = Path(__file__).resolve()
        project_root = current_file.parents[2]
        full_path = project_root / download_path

        if full_path.exists():
            df = pd.read_parquet(full_path, engine='fastparquet')
        else:
            raise ValueError(
                f"Файл не найден: {full_path.exists()}"
            )

        df = df[df['symbol'].isin(self.symbols)].reset_index(drop=True)

        if df.empty:
            raise ValueError(
                "После фильтрации по symbols датафрейм оказался пустым"
            )

        if analyse_data:
            print('\t\t\tСтатистика без фильтрации\n')
            self.statistics_df(df)

        if cleaning:
            minimal_ts_for_alignment = (
                df.groupby('symbol')['timestamp'].min().max()
            )
            df = df[df['timestamp'] >= minimal_ts_for_alignment]

            if 'turnover' in df.columns:
                df = df.drop(columns=['turnover'])

            df = df.reset_index(drop=True)

            if analyse_data:
                print(
                    '\n\n\t\t\tСтатистика после фильтрации (удаление столбца \'turnover\' и выравнивание временных рядов по активам)\n')
                self.statistics_df(df)

        return df