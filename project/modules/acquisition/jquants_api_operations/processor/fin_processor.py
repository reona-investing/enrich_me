import pandas as pd
import numpy as np
from utils.paths import Paths
import yaml
from acquisition.jquants_api_operations.processor.formatter import Formatter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.processor.code_replacement_info import codes_to_merge_dict, manual_adjustment_dict_list, codes_to_replace_dict


class FinProcessor:
    def __init__(self,
                 raw_path: str = Paths.RAW_STOCK_FIN_PARQUET,
                 processing_path: str = Paths.STOCK_FIN_PARQUET) -> None: # 財務情報の加工
        '''
        raw_stock_finを、機械学習に使える形に加工します。
        Args:
            raw_path (str): 生の財務データが保存されているパス
            processing_path (str): 加工後の財務データを保存するパス
        '''

        self.columns_info = self._load_config_yaml()
        stock_fin = self._load_fin_data(raw_path)
        stock_fin = self._filter_fin_data(stock_fin)
        stock_fin = self._rename_columns(stock_fin)
        stock_fin = self._format_dtypes(stock_fin)
        stock_fin = self._drop_duplicated_data(stock_fin)
        stock_fin = self._calculate_additional_fins(stock_fin) # 追加の要素を算出
        stock_fin = self._process_merger(stock_fin) # 合併処理を実施
        stock_fin = self._finalize_df(stock_fin)
        FileHandler.write_parquet(stock_fin, Paths.STOCK_FIN_PARQUET)

    def _load_config_yaml(self, yaml_path: str | None = None):
        if yaml_path is None:
            yaml_path = Paths.STOCK_FIN_COLUMNS_YAML
        with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
            columns_info = yaml.safe_load(yaml_file)
            columns_info = [col for col in columns_info['columns'] if col['include']]
        return columns_info

    def _load_fin_data(self, raw_path: str) -> pd.DataFrame:
        '''生の財務データを読み込みます。'''
        fin_df =  FileHandler.read_parquet(raw_path)
        return fin_df.replace('', np.nan)

    def _filter_fin_data(self, fin_df: str) -> pd.DataFrame:
        '''生の財務データを読み込みます。'''
        return fin_df[[col['raw_name'] for col in self.columns_info]]

    def _rename_columns(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        rename_dict = {col['raw_name']: col['processed_name'] \
                       for col in self.columns_info if col['raw_name'] != col['processed_name']}
        return fin_df.rename(columns=rename_dict)

    def _format_dtypes(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        '''各カラムに適切なデータ型を設定します。'''
        convert_dict_except_datetime = {col['processed_name']: eval(col['processed_dtype']) \
                                        for col in self.columns_info if col['processed_dtype'] != 'datetime'} 
        datetime_columns = [col['processed_name'] \
                            for col in self.columns_info if col['processed_dtype'] == 'datetime']

        fin_df[[x for x in convert_dict_except_datetime.keys()]] = \
            fin_df[[x for x in convert_dict_except_datetime.keys()]].astype(convert_dict_except_datetime)

        for column in datetime_columns:
            fin_df[column]= fin_df[column].astype(str).str[:10]
            fin_df[column] = pd.to_datetime(fin_df[column])
    
        code_col = self._get_column_name('LocalCode')
        fin_df[code_col] = Formatter.format_stock_code(fin_df[code_col])
        fin_df[code_col] = fin_df[code_col].replace(codes_to_replace_dict)
        return fin_df



    def _drop_duplicated_data(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        '''同一日に同一銘柄の複数レコードが存在する場合、重複を削除します。'''
        # 発表の訂正などで、同一日に複数回のリリースがある場合がある。当日最後の発表を最新の発表として残す。
        type_of_document_col = self._get_column_name('TypeOfDocument')
        disclosed_time_col = self._get_column_name('DisclosedTime')
        code_col = self._get_column_name('LocalCode')
        date_col = self._get_column_name('DisclosedDate')
        
        fin_df = fin_df.loc[~fin_df[type_of_document_col].isin(
            ['DividendForecastRevision', 'EarnForecastRevision', 'REITDividendForecastRevision', 'REITEarnForecastRevision'])]
        return fin_df.sort_values(disclosed_time_col).drop_duplicates(subset=[code_col, date_col], keep="last") #コードと開示日が重複しているデータを、最後を残して削除


    def _calculate_additional_fins(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        '''追加の財務要素を算出します。'''
        treasury_stock_fiscal_year_end_col = self._get_column_name('NumberOfTreasuryStockAtTheEndOfFiscalYear')
        shares_fiscal_year_end_col = \
            self._get_column_name('NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock')
        fiscal_year_end_date_col = self._get_column_name('CurrentFiscalYearEndDate')
        type_of_current_period_col = self._get_column_name('TypeOfCurrentPeriod')
        #期末発行済株式数の算出
        fin_df['OutstandingShares'] = np.nan
        fin_df.loc[fin_df[treasury_stock_fiscal_year_end_col].notnull(), 'OutstandingShares'] = \
            fin_df[shares_fiscal_year_end_col] - fin_df[treasury_stock_fiscal_year_end_col]
        fin_df.loc[fin_df[treasury_stock_fiscal_year_end_col].isnull(), 'OutstandingShares'] = \
            fin_df[shares_fiscal_year_end_col]
        #その他、追加で必要な列を算出
        fin_df["CurrentFiscalYear"] = fin_df[fiscal_year_end_date_col].dt.strftime("%Y")
        fin_df["ForecastFiscalYearEndDate"] = fin_df[fiscal_year_end_date_col].dt.strftime("%Y/%m")
        fin_df.loc[fin_df[type_of_current_period_col] == "FY", 'ForecastFiscalYearEndDate'] = (
            fin_df.loc[fin_df[type_of_current_period_col] == "FY", fiscal_year_end_date_col] \
            + pd.offsets.DateOffset(years=1)
                ).dt.strftime("%Y/%m")
        return fin_df

    def _process_merger(self, stock_fin:pd.DataFrame) -> pd.DataFrame:
        '''企業合併時の財務情報合成処理を行います'''
        #合併前の各社のデータを足し合わせる項目
        plus_when_merging = [col['processed_name'] for col in self.columns_info if col['plus_for_merger'] == True]
        code_col = self._get_column_name('LocalCode')
        date_col = self._get_column_name('DisclosedDate')
        #合併リストの中身の分だけ処理を繰り返し
        for key, value in codes_to_merge_dict.items():
            merger1 = stock_fin.loc[stock_fin[code_col]==value['Code1']].sort_values(date_col) #合併前1（存続）
            merger2 = stock_fin.loc[stock_fin[code_col]==value['Code2']].sort_values(date_col) #合併前2（消滅）
            #存続会社のデータを、合併前後に分ける。
            merger1_before = merger1[merger1[date_col]<=value['MergerDate']]
            merger1_after = merger1[merger1[date_col]>value['MergerDate']]
            #インデックスを揃える
            merger1_before = \
                pd.merge(merger1_before, merger2[date_col], how='outer', on=date_col).sort_values(date_col).reset_index(drop=True)
            merger2 = \
                pd.merge(merger2, merger1_before[date_col], how='outer', on=date_col).sort_values(date_col).reset_index(drop=True)
            #NaN値を埋める
            merger1_before = merger1_before.ffill()
            merger1_before = merger1_before.bfill()
            merger2 = merger2.ffill()
            merger2 = merger2.bfill()
            #merger1_beforeの値を書き換えていく。
            merger1_before[code_col] = key
            merger1_before['OutstandingShares'] = merger1_before['OutstandingShares'] + \
                                                merger2['OutstandingShares'] * value['ExchangeRate']
            merger1_before[plus_when_merging] =  merger1_before[plus_when_merging] + merger2[plus_when_merging]
            #合併前後のデータを結合する。
            merged = pd.concat([merger1_before, merger1_after], axis=0).sort_values(date_col)
            stock_fin = stock_fin[(stock_fin[code_col]!=value['Code1'])&(stock_fin[code_col]!=value['Code2'])]
            return pd.concat([stock_fin, merged], axis=0).sort_values([date_col, code_col])

    def _finalize_df(self, stock_fin: pd.DataFrame) -> pd.DataFrame:
        '''データフレームに最終処理を施します。'''
        disclosed_time_col = self._get_column_name('DisclosedTime')
        return stock_fin.drop([disclosed_time_col], axis=1).drop_duplicates(keep='last').reset_index(drop=True) # データフレームの最終処理


    def _get_column_name(self, raw_name: str) -> str:
        return next((col['processed_name'] for col in self.columns_info if col['raw_name'] == raw_name), None)


if __name__ == '__main__':
    FinProcessor()