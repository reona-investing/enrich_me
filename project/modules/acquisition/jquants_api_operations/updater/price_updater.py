import pandas as pd
from datetime import datetime
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from acquisition.jquants_api_operations.utils import FileHandler
from utils.jquants_api_utils import cli

class PriceUpdater:
    def __init__(self,
                 basic_path: str = Paths.RAW_STOCK_PRICE_PARQUET,
                 cols_yaml_path: str = Paths.RAW_STOCK_PRICE_COLUMNS_YAML,
                 ) -> None:
        """
        株価情報を更新し、指定されたパスにParquet形式で保存します。
        :param basic_path: パスのテンプレート（例: "path_to_data/0000.parquet"）
        """
        self._load_column_configs(cols_yaml_path)
        self._needs_processing = False

        current_year = datetime.today().year
        prev_year = current_year - 1

        current_year_path = self._generate_file_path(current_year, basic_path)
        prev_year_path = self._generate_file_path(prev_year, basic_path)

        # 今年のデータが存在しない場合、前年データを更新
        if not FileHandler.file_exists(current_year_path):
            self._update_yearly_stock_price(prev_year, prev_year_path)

        # 今年のデータを取得・更新
        raw_stock_price = self._update_yearly_stock_price(current_year, current_year_path)

        print(raw_stock_price.tail(2))


    def _load_column_configs(self, cols_yaml_path: str):
        ccg = ColumnConfigsGetter(cols_yaml_path)
        self.cols = {'日付': ccg.get_column_name('日付'),
                     '銘柄コード': ccg.get_column_name('銘柄コード'),
                     '調整係数': ccg.get_column_name('調整係数'),}


    def _generate_file_path(self, year: int, basic_path: str) -> str:
        """指定された年に基づいてファイルパスを生成します。"""
        return basic_path.replace('0000', str(year))


    def _update_yearly_stock_price(self, year: int, yearly_path: str) -> pd.DataFrame:
        """
        特定の年の株価情報を取得し、更新します。
        :param year: 対象の年
        :param yearly_path: 年ごとのファイルパス
        """
        # ファイルが存在すれば読み込み
        existing_data = FileHandler.read_parquet(yearly_path) if FileHandler.file_exists(yearly_path) else pd.DataFrame()

        # データ更新
        updated_data = self._update_yearly_price(year, existing_data)

        # 重複排除とクリーンアップ
        cleaned_data = updated_data.drop_duplicates().dropna()

        # 更新されたデータを保存
        FileHandler.write_parquet(cleaned_data, yearly_path)
        return cleaned_data


    def _update_yearly_price(self, year: int, existing_data: pd.DataFrame) -> pd.DataFrame:
        """
        年次データを取得し、既存データを更新します。
        :param year: 対象の年
        :param existing_data: 既存の価格情報データ
        """
        if existing_data.empty:
            fetched_data = self._fetch_full_year_stock_price(year)
            self._set_adjustment_flag(fetched_data)
            return fetched_data

        last_exist_date = pd.to_datetime(existing_data[self.cols['日付']]).iat[-1].date()
        if last_exist_date != datetime.today().date():
            new_data = self._fetch_new_stock_price(last_exist_date)
            self._set_adjustment_flag(new_data)
            return self._update_raw_stock_price(existing_data, new_data)

        return existing_data


    def _fetch_full_year_stock_price(self, year: int) -> pd.DataFrame:
        """指定された年の全期間の価格情報を取得します。"""
        return cli.get_price_range(
            start_dt=datetime(year, 1, 1),
            end_dt=datetime(year, 12, 31)
        )


    def _fetch_new_stock_price(self, last_exist_date: datetime) -> pd.DataFrame:
        """最新の日付までの新しい価格情報を取得します。"""
        return cli.get_price_range(
            start_dt=last_exist_date,
            end_dt=datetime.today()
        )


    def _set_adjustment_flag(self, fetched_stock_price: pd.DataFrame):
        """AdjustmentFactorが変更された場合のフラグを設定します。"""
        if any(fetched_stock_price[self.cols['調整係数']] != 1):
            self._needs_processing = True


    def _update_raw_stock_price(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """既存の価格情報に新しいデータを追加し、重複を削除します。"""
        combined_data = pd.concat([existing_data, new_data], axis=0)
        combined_data[self.cols['日付']] = pd.to_datetime(combined_data[self.cols['日付']])
        return combined_data[
            combined_data[self.cols['銘柄コード']].notnull()
        ].drop_duplicates(subset=[self.cols['日付'], self.cols['銘柄コード']])

    @property
    def needs_processing(self) -> bool:
        """価格データの再加工が必要かどうかを示す。"""
        return self._needs_processing


if __name__ == '__main__':
    updater = PriceUpdater()
    print(updater.needs_processing)

