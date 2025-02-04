import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
from utils.flag_manager import flag_manager, Flags
from utils.paths import Paths
from utils import yaml_utils
from acquisition.jquants_api_operations.processor.formatter import Formatter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.processor.code_replacement_info import manual_adjustment_dict_list,codes_to_replace_dict


class PriceProcessor:
    def __init__(self,
                 raw_basic_path: str = Paths.RAW_STOCK_PRICE_PARQUET,
                 processing_basic_path: str = Paths.STOCK_PRICE_PARQUET,
                 yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML):
        """
        価格情報を加工して、機械学習用に整形します。

        Args:
            raw_basic_path (str): 生の株価データが保存されているパス。
            processing_basic_path (str): 加工後の株価データを保存するパス。
        """
        self.columns_info = yaml_utils.including_columns_loader(yaml_path, 'original_columns')
        self._get_col_names()

        end_date = datetime.today()
        temp_cumprod = {}
        
        for year in range(end_date.year, 2012, -1):
            is_latest_file = year == end_date.year
            should_process = is_latest_file or flag_manager.flags[Flags.PROCESS_STOCK_PRICE]
            if should_process:
                stock_price = self._load_yearly_raw_data(raw_basic_path, year)
                if stock_price.empty:
                    continue
                stock_price, temp_cumprod = self._process_stock_price(stock_price, temp_cumprod, is_latest_file)
                self._save_yearly_data(stock_price, processing_basic_path, year)

    def _get_col_names(self):
        self.code_col = self._column_name_getter('Code')
        self.date_col = self._column_name_getter('Date')
        self.open_col = self._column_name_getter('Open')
        self.close_col = self._column_name_getter('Close')
        self.high_col = self._column_name_getter('High')
        self.low_col = self._column_name_getter('Low')
        self.volume_col = self._column_name_getter('Volume')
        self.turnover_col = self._column_name_getter('TurnoverValue')
        self.adjustment_factor_col = self._column_name_getter('AdjustmentFactor')

    # サブプロセス
    def _load_yearly_raw_data(self, raw_basic_path: str, year: int, raw_name_key: str = 'raw_name') -> pd.DataFrame:
        '''取得したままの年次株価データを読み込みます。'''
        raw_path = raw_basic_path.replace('0000', str(year))
        usecols = [col[raw_name_key] for col in self.columns_info]
        dict_for_rename = self._get_dict_for_rename(usecols)
        df = FileHandler.read_parquet(raw_path, usecols=usecols)
        return df.rename(columns=dict_for_rename)

    def _get_dict_for_rename(self, columns: list[str]) -> dict[str, str]:
        renamed_columns = self._column_names_getter(columns)
        return {x: y for x, y in zip(columns, renamed_columns)}
        

    def _process_stock_price(self, stock_price: pd.DataFrame, temp_cumprod: dict[str, float], is_latest_file: bool) -> pd.DataFrame:
        """
        価格データを加工します。
        Args:
            stock_price (pd.DataFrame): 加工前の株価データ
            temp_cumprod (dict[str, float]): 
                処理時点での銘柄ごとの暫定の累積調整係数を格納（キー: 銘柄コード、値：暫定累積調整係数）
            is_latest_file (bool): stock_priceが最新期間のファイルかどうか
        Returns:
            pd.DataFrame: 加工された株価データ。
        """
        stock_price[self.code_col] = Formatter.format_stock_code(stock_price[self.code_col].astype(str))
        stock_price[self.code_col] = self._replace_code(stock_price[self.code_col])
        stock_price = self._fill_suspension_period(stock_price)
        stock_price = self._format_dtypes(stock_price)
        stock_price = self._remove_system_failure_day(stock_price)
        stock_price, temp_cumprod = self._apply_cumulative_adjustment_factor(
            stock_price, temp_cumprod, is_latest_file, manual_adjustment_dict_list
        )
        return self._finalize_price_data(stock_price), temp_cumprod

    def _replace_code(self, code_column: pd.Series) -> pd.Series:
        """ルールに従い、銘柄コードを置換します。"""
        return code_column.replace(codes_to_replace_dict)

    def _fill_suspension_period(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """銘柄コード変更前後の欠損期間のデータを埋めます。"""
        rows_to_add = []
        date_list = stock_price[self.date_col].unique()
        codes_after_replacement = codes_to_replace_dict.values()

        for code_replaced in codes_after_replacement:
            dates_to_fill = self._get_missing_dates(stock_price, code_replaced, date_list)
            rows_to_add.extend(self._create_missing_rows(stock_price, code_replaced, dates_to_fill))

        return pd.concat([stock_price] + rows_to_add, axis=0)

    def _get_missing_dates(self, stock_price: pd.DataFrame, code_replaced: str, date_list: List[str]) -> List[str]:
        """データが欠損している日付を取得します。"""
        existing_dates = stock_price.loc[stock_price[self.code_col] == code_replaced, self.date_col].unique()
        return [x for x in date_list if x not in existing_dates]

    def _create_missing_rows(self, stock_price: pd.DataFrame, code: str, dates_to_fill: List[str]) -> List[pd.DataFrame]:
        """欠損期間の行を作成します。"""
        rows = []
        if len(dates_to_fill) <= 5:
            for date in dates_to_fill:
                last_date = stock_price.loc[(stock_price[self.code_col] == code) & (stock_price[self.date_col] <= self.date_col), self.date_col].max()
                value_to_fill = stock_price.loc[(stock_price[self.code_col] == code) & (stock_price[self.date_col] == last_date), self.close_col].values[0]
                row_to_add = {self.date_col: date, self.code_col: code, self.open_col: value_to_fill, self.close_col: value_to_fill,
                            self.high_col: value_to_fill, self.low_col: value_to_fill, self.volume_col: 0,
                            self.turnover_col: 0, self.adjustment_factor_col: 1}
                rows.append(pd.DataFrame([row_to_add], index=[0]))
        return rows

    def _format_dtypes(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """データ型を整形します。"""
        return yaml_utils.dtypes_converter(self.columns_info, stock_price)

    def _remove_system_failure_day(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """システム障害によるデータ欠損日を除外します。"""
        return stock_price[stock_price[self.date_col] != '2020-10-01']

    def _apply_cumulative_adjustment_factor(
            self, 
            stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool,
            manual_adjustment_dict_list: List[Dict]
            ) -> Tuple[pd.DataFrame, dict]:
        """価格データ（OHLCV）に累積調整係数を適用します。"""
        stock_price = stock_price.sort_values([self.code_col, self.date_col]).set_index(self.code_col, drop=True)
        stock_price = stock_price.groupby(self.code_col, group_keys=False).apply(self._calculate_cumulative_adjustment_factor).reset_index(drop=False)

        if not is_latest_file:
            stock_price = self._inherit_cumulative_values(stock_price, temp_cumprod)

        temp_cumprod = stock_price.groupby(self.code_col)['CumulativeAdjustmentFactor'].first().to_dict() # 遡っていくため、初日の値を参照
        stock_price = self._apply_manual_adjustments(stock_price, manual_adjustment_dict_list)

        for col in [self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col]:
            stock_price[col] /= stock_price['CumulativeAdjustmentFactor']

        return stock_price, temp_cumprod

    def _calculate_cumulative_adjustment_factor(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """累積調整係数を計算します。"""
        stock_price = stock_price.sort_values(self.date_col, ascending=False)
        stock_price[self.adjustment_factor_col] = stock_price[self.adjustment_factor_col].shift(1).fillna(1.0)
        stock_price['CumulativeAdjustmentFactor'] = 1 / stock_price[self.adjustment_factor_col].cumprod()
        return stock_price.sort_values(self.date_col)

    def _inherit_cumulative_values(self, stock_price: pd.DataFrame, temp_cumprod: dict) -> pd.DataFrame:
        """計算途中の暫定累積調整係数を引き継ぎます。"""
        stock_price['InheritedValue'] = stock_price[self.code_col].map(temp_cumprod).fillna(1)
        stock_price['CumulativeAdjustmentFactor'] *= stock_price['InheritedValue']
        return stock_price.drop(columns='InheritedValue')

    def _apply_manual_adjustments(self, stock_price: pd.DataFrame, manual_adjustments: List[Dict]) -> pd.DataFrame:
        """元データで未掲載の株式分割・併合について、累積調整係数をマニュアルで調整する"""
        for adjustment in manual_adjustments:
            condition = (stock_price[self.code_col] == adjustment[self.code_col]) & (stock_price[self.date_col] < adjustment[self.date_col])
            stock_price.loc[condition, 'CumulativeAdjustmentFactor'] *= adjustment['Rate']
        return stock_price

    def _finalize_price_data(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """最終的なデータ整形を行う。"""
        stock_price = stock_price.dropna(subset=[self.code_col])
        stock_price = stock_price.drop_duplicates(subset=[self.date_col, self.code_col], keep='last')
        return stock_price.sort_values([self.date_col, self.code_col]).reset_index(drop=True)[
            [self.date_col, self.code_col, self.open_col, self.high_col, self.low_col, self.close_col, self.volume_col, self.adjustment_factor_col, 'CumulativeAdjustmentFactor', self.turnover_col]
        ]

    def _save_yearly_data(self, df: pd.DataFrame, processing_basic_path: str, year: int) -> None:
        '''年次の加工後価格データを保存する。'''
        save_path = processing_basic_path.replace('0000', str(year))
        FileHandler.write_parquet(df, save_path)

    # --------------------------------------------------------------------------
    #  以下、ヘルパーメソッド
    # --------------------------------------------------------------------------

    def _column_name_getter(self, raw_name: str) -> str:
        """
        指定したカラム名の変換後の名称を取得。

        Args:
            raw_name (str): 変換前のカラム名

        Returns:
            str: 変換後のカラム名
        """ 
        return yaml_utils.column_name_getter(self.columns_info, {'raw_name': raw_name}, 'processed_name')


    def _column_names_getter(self, raw_names: list[str]) -> list[str]:
        """
        指定したカラム名（複数）の変換後の名称を一括で取得。

        Args:
            raw_names (list[str]): 変換前のカラム名一覧を一括で取得

        Returns:
            list[str]: 変換後のカラム名を格納したリスト
        """
        return [yaml_utils.column_name_getter(self.columns_info, {'raw_name': raw_name}, 'processed_name') for raw_name in raw_names]


if __name__ == '__main__':

    flag_manager.set_flag(Flags.PROCESS_STOCK_PRICE, True)
    PriceProcessor()