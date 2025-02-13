import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Callable
from utils.flag_manager import flag_manager, Flags
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter

from acquisition.jquants_api_operations.processor.formatter import Formatter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.processor.code_replacement_info import manual_adjustment_dict_list,codes_to_replace_dict


class PriceProcessor:
    def __init__(self,
                 raw_basic_path: str = Paths.RAW_STOCK_PRICE_PARQUET,
                 processing_basic_path: str = Paths.STOCK_PRICE_PARQUET,
                 raw_cols_yaml_path: str = Paths.RAW_STOCK_PRICE_COLUMNS_YAML,
                 cols_yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML):
        """
        価格情報を加工して、機械学習用に整形します。

        Args:
            raw_basic_path (str): 生の株価データが保存されているパス。
            processing_basic_path (str): 加工後の株価データを保存するパス。
        """
        self.columns_info = self._get_col_info(raw_cols_yaml_path, cols_yaml_path)

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

    def _get_col_info(self, raw_cols_yaml_path: str, cols_yaml_path: str):
        raw_ccg = ColumnConfigsGetter(raw_cols_yaml_path)
        ccg = ColumnConfigsGetter(cols_yaml_path)
        keys = ['日付', '銘柄コード', '始値', '終値', '高値', '安値', '取引高', '取引代金', '調整係数']
        self.raw_cols = self._get_column_mapping(keys, raw_ccg.get_column_name)
        self.cols = self._get_column_mapping(keys, ccg.get_column_name)
        self.col_dtypes = self._get_column_mapping(keys, ccg.get_column_dtype)

    def _get_column_mapping(self, keys, column_config_getter: Callable):
        """
        指定されたキーリストに基づいて、ColumnConfigsGetter からカラム名のマッピング辞書を作成する。

        Parameters:
            keys (list): 取得するキーのリスト
            column_config_getter (ColumnConfigsGetterの関数): カラム名を取得用の関数

        Returns:
            dict: 指定されたキーと取得したカラム名の辞書
        """
        return {key: column_config_getter(key) for key in keys if column_config_getter(key) is not None}

    # サブプロセス
    def _load_yearly_raw_data(self, raw_basic_path: str, year: int) -> pd.DataFrame:
        '''取得したままの年次株価データを読み込みます。'''
        raw_path = raw_basic_path.replace('0000', str(year))
        usecols = self.raw_cols.values()
        dict_for_rename = self._get_dict_for_rename()
        df = FileHandler.read_parquet(raw_path, usecols=usecols)
        return df.rename(columns=dict_for_rename)


    def _get_dict_for_rename(self) -> dict[str, str]:
        return {self.raw_cols[key]: self.cols[key] for key in self.raw_cols.keys()}
        

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
        stock_price[self.cols['銘柄コード']] = Formatter.format_stock_code(stock_price[self.cols['銘柄コード']].astype(str))
        stock_price[self.cols['銘柄コード']] = self._replace_code(stock_price[self.cols['銘柄コード']])
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
        date_list = stock_price[self.cols['日付']].unique()
        codes_after_replacement = codes_to_replace_dict.values()

        for code_replaced in codes_after_replacement:
            dates_to_fill = self._get_missing_dates(stock_price, code_replaced, date_list)
            rows_to_add.extend(self._create_missing_rows(stock_price, code_replaced, dates_to_fill))

        return pd.concat([stock_price] + rows_to_add, axis=0)

    def _get_missing_dates(self, stock_price: pd.DataFrame, code_replaced: str, date_list: List[str]) -> List[str]:
        """データが欠損している日付を取得します。"""
        existing_dates = stock_price.loc[stock_price[self.cols['銘柄コード']] == code_replaced, self.cols['日付']].unique()
        return [x for x in date_list if x not in existing_dates]

    def _create_missing_rows(self, stock_price: pd.DataFrame, code: str, dates_to_fill: List[str]) -> List[pd.DataFrame]:
        """欠損期間の行を作成します。"""
        rows = []
        if len(dates_to_fill) <= 5:
            for date in dates_to_fill:
                last_date = \
                    stock_price.loc[(stock_price[self.cols['銘柄コード']] == code) & (stock_price[self.cols['日付']] <= self.cols['日付']), self.cols['日付']].max()
                value_to_fill = \
                    stock_price.loc[(stock_price[self.cols['銘柄コード']] == code) & (stock_price[self.cols['日付']] == last_date), self.cols['終値']].values[0]
                row_to_add = {self.cols['日付']: date, self.cols['銘柄コード']: code, self.cols['始値']: value_to_fill, self.cols['終値']: value_to_fill,
                            self.cols['高値']: value_to_fill, self.cols['安値']: value_to_fill, self.cols['取引高']: 0,
                            self.cols['取引代金']: 0, self.cols['調整係数']: 1}
                rows.append(pd.DataFrame([row_to_add], index=[0]))
        return rows

    def _format_dtypes(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """データ型を整形します。"""
        type_mapping = {self.cols[key]: eval(self.col_dtypes[key]) for key in self.cols.keys()
                        if self.col_dtypes[key] != 'datetime'} 
        datetime_columns = [self.cols[key] for key in self.cols.keys() if self.col_dtypes[key] == 'datetime']

        stock_price[[x for x in type_mapping.keys()]] = stock_price[[x for x in type_mapping.keys()]].astype(type_mapping)

        for column in datetime_columns:
            stock_price[column]= stock_price[column].astype(str).str[:10]
            stock_price[column] = pd.to_datetime(stock_price[column])

        return stock_price

    def _remove_system_failure_day(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """システム障害によるデータ欠損日を除外します。"""
        return stock_price[stock_price[self.cols['日付']] != '2020-10-01']

    def _apply_cumulative_adjustment_factor(
            self, 
            stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool,
            manual_adjustment_dict_list: List[Dict]
            ) -> Tuple[pd.DataFrame, dict]:
        """価格データ（OHLCV）に累積調整係数を適用します。"""
        stock_price = stock_price.sort_values([self.cols['銘柄コード'], self.cols['日付']]).set_index(self.cols['銘柄コード'], drop=True)
        stock_price = stock_price.groupby(self.cols['銘柄コード'], group_keys=False).apply(self._calculate_cumulative_adjustment_factor).reset_index(drop=False)

        if not is_latest_file:
            stock_price = self._inherit_cumulative_values(stock_price, temp_cumprod)

        temp_cumprod = stock_price.groupby(self.cols['銘柄コード'])['CumulativeAdjustmentFactor'].first().to_dict() # 遡っていくため、初日の値を参照
        stock_price = self._apply_manual_adjustments(stock_price, manual_adjustment_dict_list)

        for col in [self.cols['始値'], self.cols['高値'], self.cols['安値'], self.cols['終値'], self.cols['取引高']]:
            stock_price[col] /= stock_price['CumulativeAdjustmentFactor']

        return stock_price, temp_cumprod

    def _calculate_cumulative_adjustment_factor(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """累積調整係数を計算します。"""
        stock_price = stock_price.sort_values(self.cols['日付'], ascending=False)
        stock_price[self.cols['調整係数']] = stock_price[self.cols['調整係数']].shift(1).fillna(1.0)
        stock_price['CumulativeAdjustmentFactor'] = 1 / stock_price[self.cols['調整係数']].cumprod()
        return stock_price.sort_values(self.cols['日付'])

    def _inherit_cumulative_values(self, stock_price: pd.DataFrame, temp_cumprod: dict) -> pd.DataFrame:
        """計算途中の暫定累積調整係数を引き継ぎます。"""
        stock_price['InheritedValue'] = stock_price[self.cols['銘柄コード']].map(temp_cumprod).fillna(1)
        stock_price['CumulativeAdjustmentFactor'] *= stock_price['InheritedValue']
        return stock_price.drop(columns='InheritedValue')

    def _apply_manual_adjustments(self, stock_price: pd.DataFrame, manual_adjustments: List[Dict]) -> pd.DataFrame:
        """元データで未掲載の株式分割・併合について、累積調整係数をマニュアルで調整する"""
        for adjustment in manual_adjustments:
            condition = \
                (stock_price[self.cols['銘柄コード']] == adjustment[self.cols['銘柄コード']]) & (stock_price[self.cols['日付']] < adjustment[self.cols['日付']])
            stock_price.loc[condition, 'CumulativeAdjustmentFactor'] *= adjustment['Rate']
        return stock_price

    def _finalize_price_data(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """最終的なデータ整形を行う。"""
        stock_price = stock_price.dropna(subset=[self.cols['銘柄コード']])
        stock_price = stock_price.drop_duplicates(subset=[self.cols['日付'], self.cols['銘柄コード']], keep='last')
        return stock_price.sort_values([self.cols['日付'], self.cols['銘柄コード']]).reset_index(drop=True)[
            [self.cols['日付'], 
             self.cols['銘柄コード'], 
             self.cols['始値'], 
             self.cols['高値'], 
             self.cols['安値'], 
             self.cols['終値'], 
             self.cols['取引高'], 
             self.cols['調整係数'], 
             'CumulativeAdjustmentFactor', 
             self.cols['取引代金']]
        ]

    def _save_yearly_data(self, df: pd.DataFrame, processing_basic_path: str, year: int) -> None:
        '''年次の加工後価格データを保存する。'''
        save_path = processing_basic_path.replace('0000', str(year))
        FileHandler.write_parquet(df, save_path)



if __name__ == '__main__':
    flag_manager.set_flag(Flags.PROCESS_STOCK_PRICE, True)
    PriceProcessor()