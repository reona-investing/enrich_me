import pandas as pd
from datetime import datetime
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from utils.jquants_api_utils import cli
from acquisition.jquants_api_operations.utils import FileHandler


class FinUpdater:
    def __init__(self, 
                 path: str = Paths.RAW_STOCK_FIN_PARQUET,
                 cols_yaml_path: str = Paths.RAW_STOCK_FIN_COLUMNS_YAML,
                 ) -> None:
        """
        財務情報の更新を行い、指定されたパスにParquet形式で保存する。
        :param path: Parquetファイルの保存先パス
        """
        self._load_column_configs(cols_yaml_path)
        existing_data = self._load_existing_data(path)
        fetched_data = self._fetch_data(existing_data)
        merged_data = self._merge(existing_data, fetched_data, key=self.cols['開示番号'])
        formatted_data = self._format(merged_data)
        
        print(formatted_data.tail(2))
        FileHandler.write_parquet(formatted_data, path)


    def _load_column_configs(self, cols_yaml_path: str):
        ccg = ColumnConfigsGetter(cols_yaml_path)
        assert ccg.get_column_name('日付') is not None '"日付"がyamlに定義されていません。'
        cols = {'日付': ccg.get_column_name('日付'),
                     '開示番号': ccg.get_column_name('開示番号'),}
        # None チェック
        for key, value in cols.items():
            if value is None:
                raise ValueError(f"列名 '{key}' に対応する値が None です。ccg.get_column_name('{key}') の戻り値を確認してください。")
        self.cols = cols

    def _load_existing_data(self, path: str) -> pd.DataFrame:
        """既存データを読み込む。ファイルが存在しない場合は空のDataFrameを返す。"""
        if FileHandler.file_exists(path):
            return FileHandler.read_parquet(path)
        return pd.DataFrame()


    def _fetch_data(self, existing_data: pd.DataFrame) -> pd.DataFrame:
        """財務情報をAPIから取得する。"""
        start_date = datetime(2000, 1, 1)
        if not existing_data.empty:
            start_date = pd.to_datetime(existing_data[self.cols['日付']].iat[-1])
        end_date = datetime.today()
        return cli.get_statements_range(start_dt=start_date, end_dt=end_date)


    def _merge(self, existing_data: pd.DataFrame, new_data: pd.DataFrame, key: str) -> pd.DataFrame:
        """既存データと新規データを結合し、重複を排除する。"""
        if existing_data.empty:
            return new_data
        unique_existing = existing_data.loc[~existing_data[key].isin(new_data[key])]
        return pd.concat([new_data, unique_existing], axis=0).reset_index(drop=True)


    def _format(self, data: pd.DataFrame) -> pd.DataFrame:
        """データを指定された形式に整形する。"""
        return data.astype(str).sort_values(self.cols['開示番号']).reset_index(drop=True)


if __name__ == '__main__':
    FinUpdater()
