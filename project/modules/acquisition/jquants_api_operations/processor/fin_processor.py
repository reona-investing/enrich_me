import pandas as pd
import numpy as np
from utils.paths import Paths
from utils import yaml_utils
from acquisition.jquants_api_operations.processor.formatter import Formatter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.processor.code_replacement_info import codes_to_merge_dict, manual_adjustment_dict_list, codes_to_replace_dict

class FinProcessor:
    """
    財務データの前処理を行うクラス。
    
    Jquants APIで取得した財務情報データを機械学習に適した形式に変換し、整形・加工・合併処理を行う。
    加工後のデータは Parquet 形式で保存する。
    """
    def __init__(self,
                 raw_path: str = Paths.RAW_STOCK_FIN_PARQUET,
                 processing_path: str = Paths.STOCK_FIN_PARQUET,
                 yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML) -> None: # 財務情報の加工
        """
        インスタンス生成と同時にデータの加工と保存を行う。

        Args:
            raw_path (str): 生の財務データの保存パス
            processing_path (str): 加工後の財務データの保存パス
            yaml_path (str): YAML設定ファイルのパス
        """

        self.original_columns_info = self._load_yaml_original_columns(yaml_path, 'original_columns')
        self.calculated_columns_info = self._load_yaml_original_columns(yaml_path, 'calculated_columns')
        stock_fin = self._get_col_names()
        stock_fin = self._load_fin_data(raw_path)
        stock_fin = self._filter_fin_data(stock_fin)
        stock_fin = self._rename_columns(stock_fin)
        stock_fin = self._format_dtypes(stock_fin)
        stock_fin = self._drop_duplicated_data(stock_fin)
        stock_fin = self._calculate_additional_fins(stock_fin)
        stock_fin = self._process_merger(stock_fin)
        stock_fin = self._finalize_df(stock_fin)
        FileHandler.write_parquet(stock_fin, processing_path)

    def _load_yaml_original_columns(self, yaml_path: str, key: str):
        """
        財務データのカラム情報を YAML からロードし、使用するカラムのみを抽出する。

        Args:
            yaml_path (str | None): YAML ファイルのパス
            key (str): YAMLファイル内の大元のキー

        Returns:
            list[dict]: 設定されたカラム情報のリスト
        """
        return yaml_utils.including_columns_loader(yaml_path, key)
    
    def _get_col_names(self):
        self.code_col = self._column_name_getter('LocalCode')
        self.date_col = self._column_name_getter('DisclosedDate')
        self.type_of_document_col = self._column_name_getter('TypeOfDocument')
        self.disclosed_time_col = self._column_name_getter('DisclosedTime')
        self.treasury_stock_fiscal_year_end_col = self._column_name_getter('NumberOfTreasuryStockAtTheEndOfFiscalYear')
        self.shares_fiscal_year_end_col = \
            self._column_name_getter('NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock')
        self.fiscal_year_end_date_col = self._column_name_getter('CurrentFiscalYearEndDate')
        self.type_of_current_period_col = self._column_name_getter('TypeOfCurrentPeriod')
        self.outstanding_shares_col = yaml_utils.column_name_getter(self.calculated_columns_info, {'name': 'OUTSTANDING_SHARES'}, 'col_name')
        self.fiscal_year_col = yaml_utils.column_name_getter(self.calculated_columns_info, {'name': 'CURRENT_FISCAL_YEAR'}, 'col_name')
        self.forecast_fy_end_date_col = yaml_utils.column_name_getter(self.calculated_columns_info, {'name': 'FORECAST_FISCAL_YEAR_END_DATE'}, 'col_name')


    def _load_fin_data(self, raw_path: str) -> pd.DataFrame:
        """
        生の財務データを読み込み、空白値を NaN に変換する。

        Args:
            raw_path (str): 財務データのファイルパス

        Returns:
            pd.DataFrame: 読み込んだ財務データ
        """
        fin_df =  FileHandler.read_parquet(raw_path)
        return fin_df.replace('', np.nan)

    def _filter_fin_data(self, fin_df: str) -> pd.DataFrame:
        """
        設定ファイルをもとに、必要な列のみを抽出。

        Args:
            fin_df (pd.DataFrame): 読み込んだ財務データ

        Returns:
            pd.DataFrame: フィルタリング後の財務データ
        """
        return fin_df[[col['raw_name'] for col in self.original_columns_info]]

    def _rename_columns(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        設定ファイルをもとにカラム名を変換する。

        Args:
            fin_df (pd.DataFrame): フィルタリング後の財務データ

        Returns:
            pd.DataFrame: カラム名変換後の財務データ
        """
        rename_dict = {col['raw_name']: col['processed_name'] \
                       for col in self.original_columns_info if col['raw_name'] != col['processed_name']}
        return fin_df.rename(columns=rename_dict)

    def _format_dtypes(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        設定ファイルをもとに各カラムに適切なデータ型を設定。

        Args:
            fin_df (pd.DataFrame): カラム名変換後の財務データ

        Returns:
            pd.DataFrame: データ型変換後の財務データ
        """
        fin_df = yaml_utils.dtypes_converter(self.original_columns_info, fin_df)
    
        fin_df[self.code_col] = Formatter.format_stock_code(fin_df[self.code_col])
        fin_df[self.code_col] = fin_df[self.code_col].replace(codes_to_replace_dict)
        return fin_df

    def _drop_duplicated_data(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        重複データを削除し、最新の発表データを保持する。

        Args:
            fin_df (pd.DataFrame): データ型変換後の財務データ

        Returns:
            pd.DataFrame: 重複削除後の財務データ
        """
        # 発表の訂正などで、同一日に複数回のリリースがある場合がある。当日最後の発表を最新の発表として残す。
        fin_df = fin_df.loc[~fin_df[self.type_of_document_col].isin(
            ['DividendForecastRevision', 'EarnForecastRevision', 'REITDividendForecastRevision', 'REITEarnForecastRevision'])]
        return fin_df.sort_values(self.disclosed_time_col).drop_duplicates(subset=[self.code_col, self.date_col], keep="last") #コードと開示日が重複しているデータを、最後を残して削除

    def _calculate_additional_fins(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        追加の財務指標を計算。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        #期末発行済株式数の算出
        fin_df[self.outstanding_shares_col] = np.nan
        fin_df.loc[fin_df[self.treasury_stock_fiscal_year_end_col].notnull(), self.outstanding_shares_col] = \
            fin_df[self.shares_fiscal_year_end_col] - fin_df[self.treasury_stock_fiscal_year_end_col]
        fin_df.loc[fin_df[self.treasury_stock_fiscal_year_end_col].isnull(), self.outstanding_shares_col] = \
            fin_df[self.shares_fiscal_year_end_col]
        #その他、追加で必要な列を算出
        fin_df[self.fiscal_year_col] = fin_df[self.fiscal_year_end_date_col].dt.strftime("%Y")
        fin_df[self.forecast_fy_end_date_col] = fin_df[self.fiscal_year_end_date_col].dt.strftime("%Y/%m")
        fin_df.loc[fin_df[self.type_of_current_period_col] == "FY", self.forecast_fy_end_date_col] = (
            fin_df.loc[fin_df[self.type_of_current_period_col] == "FY", self.fiscal_year_end_date_col] \
            + pd.offsets.DateOffset(years=1)
                ).dt.strftime("%Y/%m")
        return fin_df

    def _process_merger(self, stock_fin:pd.DataFrame) -> pd.DataFrame:
        """
        企業合併時の財務情報の合成を行います。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        #合併前の各社のデータを足し合わせる項目
        plus_when_merging = [col['processed_name'] for col in self.original_columns_info if col['plus_for_merger'] == True]
        #合併リストの中身の分だけ処理を繰り返し
        for key, value in codes_to_merge_dict.items():
            merger1 = stock_fin.loc[stock_fin[self.code_col]==value['Code1']].sort_values(self.date_col) #合併前1（存続）
            merger2 = stock_fin.loc[stock_fin[self.code_col]==value['Code2']].sort_values(self.date_col) #合併前2（消滅）
            #存続会社のデータを、合併前後に分ける。
            merger1_before = merger1[merger1[self.date_col]<=value['MergerDate']]
            merger1_after = merger1[merger1[self.date_col]>value['MergerDate']]
            #インデックスを揃える
            merger1_before = \
                pd.merge(merger1_before, merger2[self.date_col], how='outer', on=self.date_col).sort_values(self.date_col).reset_index(drop=True)
            merger2 = \
                pd.merge(merger2, merger1_before[self.date_col], how='outer', on=self.date_col).sort_values(self.date_col).reset_index(drop=True)
            #NaN値を埋める
            merger1_before = merger1_before.ffill()
            merger1_before = merger1_before.bfill()
            merger2 = merger2.ffill()
            merger2 = merger2.bfill()
            #merger1_beforeの値を書き換えていく。
            merger1_before[self.code_col] = key
            merger1_before[self.outstanding_shares_col] = merger1_before[self.outstanding_shares_col] + \
                                                merger2[self.outstanding_shares_col] * value['ExchangeRate']
            merger1_before[plus_when_merging] =  merger1_before[plus_when_merging] + merger2[plus_when_merging]
            #合併前後のデータを結合する。
            merged = pd.concat([merger1_before, merger1_after], axis=0).sort_values(self.date_col)
            stock_fin = stock_fin[(stock_fin[self.code_col]!=value['Code1'])&(stock_fin[self.code_col]!=value['Code2'])]
            return pd.concat([stock_fin, merged], axis=0).sort_values([self.date_col, self.code_col])

    def _finalize_df(self, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """
        最終データの整形処理を行う。

        Args:
            stock_fin (pd.DataFrame): 財務データの最終処理前の状態

        Returns:
            pd.DataFrame: 最終処理後のデータ
        """
        return stock_fin.drop([self.disclosed_time_col], axis=1).drop_duplicates(keep='last').reset_index(drop=True) # データフレームの最終処理

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
        return yaml_utils.column_name_getter(self.original_columns_info, {'raw_name': raw_name}, 'processed_name')


if __name__ == '__main__':
    FinProcessor()