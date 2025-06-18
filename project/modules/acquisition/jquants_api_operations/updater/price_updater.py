import pandas as pd
from datetime import datetime
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from acquisition.jquants_api_operations.utils import FileHandler
from utils.jquants_api_utils import cli

class PriceUpdater:
    def __init__(self,
                 path: str = Paths.RAW_STOCK_PRICE_PARQUET,
                 cols_yaml_path: str = Paths.RAW_STOCK_PRICE_COLUMNS_YAML,
                 ) -> None:
        """
        株価情報を更新し、指定されたパスにParquet形式で保存します。
        :param basic_path: パスのテンプレート（例: "path_to_data/0000.parquet"）
        """
        self._load_column_configs(cols_yaml_path)

        # 今年のデータを取得・更新
        raw_stock_price = self._update_stock_price(path)
        raw_stock_price = raw_stock_price.reset_index(drop=True)
        
        FileHandler.write_parquet(raw_stock_price, path)



    def _load_column_configs(self, cols_yaml_path: str):
        ccg = ColumnConfigsGetter(cols_yaml_path)
        self.cols = {'日付': ccg.get_column_name('日付'),
                     '銘柄コード': ccg.get_column_name('銘柄コード'),
                     '調整係数': ccg.get_column_name('調整係数'),}


    def _update_stock_price(self, path: str) -> pd.DataFrame:
        """
        特定の年の株価情報を取得し、更新します。
        :param path: 価格情報データフレームのパス
        """
        existing_data = FileHandler.read_parquet(path)
        last_exist_date = pd.to_datetime(existing_data[self.cols['日付']]).iat[-1].date()

        if last_exist_date != datetime.today().date():
            new_data = self._fetch_new_stock_price(last_exist_date)
            return self._update_raw_stock_price(existing_data, new_data)

        return existing_data


    def _fetch_new_stock_price(self, last_exist_date: datetime) -> pd.DataFrame:
        """最新の日付までの新しい価格情報を取得します。"""
        return cli.get_price_range(
            start_dt=last_exist_date,
            end_dt=datetime.today()
        )

    def _update_raw_stock_price(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """既存の価格情報に新しいデータを追加し、重複を削除します。"""
        combined_data = pd.concat([existing_data, new_data], axis=0)
        combined_data[self.cols['日付']] = pd.to_datetime(combined_data[self.cols['日付']])
        return combined_data[
            combined_data[self.cols['銘柄コード']].notnull()
        ].drop_duplicates(subset=[self.cols['日付'], self.cols['銘柄コード']])


if __name__ == '__main__':
    updater = PriceUpdater()

