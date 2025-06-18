import pandas as pd
import numpy as np
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from acquisition.jquants_api_operations.processor.formatter import Formatter
from acquisition.jquants_api_operations.utils import FileHandler
from acquisition.jquants_api_operations.processor.code_replacement_info import codes_to_merge_dict, manual_adjustment_dict_list, codes_to_replace_dict
from typing import Callable

class FinProcessor:
    """
    財務データの前処理を行うクラス。
    
    Jquants APIで取得した財務情報データを機械学習に適した形式に変換し、整形・加工・合併処理を行う。
    加工後のデータは Parquet 形式で保存する。
    """
    def __init__(self,
                 raw_path: str = Paths.RAW_STOCK_FIN_PARQUET,
                 processing_path: str = Paths.STOCK_FIN_PARQUET,
                 raw_cols_yaml_path: str = Paths.RAW_STOCK_FIN_COLUMNS_YAML,
                 cols_yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML,
                 ) -> None: # 財務情報の加工
        """
        インスタンス生成と同時にデータの加工と保存を行う。

        Args:
            raw_path (str): 生の財務データの保存パス
            processing_path (str): 加工後の財務データの保存パス
            yaml_path (str): YAML設定ファイルのパス
        """

        self._get_col_info(raw_cols_yaml_path, cols_yaml_path)
        stock_fin = self._load_fin_data(raw_path)
        stock_fin = self._rename_columns(stock_fin)
        stock_fin = self._format_dtypes(stock_fin)
        stock_fin = self._drop_duplicated_data(stock_fin)
        stock_fin = self._calculate_additional_fins(stock_fin)
        stock_fin = self._process_merger(stock_fin)
        stock_fin = self._finalize_df(stock_fin)
        FileHandler.write_parquet(stock_fin, processing_path)

    def _get_col_info(self, raw_cols_yaml_path: str, cols_yaml_path: str):
        raw_ccg = ColumnConfigsGetter(raw_cols_yaml_path)
        ccg = ColumnConfigsGetter(cols_yaml_path)
        keys = ['日付', '銘柄コード', '開示書類種別', '時刻', '期末自己株式数', '期末発行済株式数', '当事業年度終了日', '当会計期間終了日', '当会計期間の種類', 
                'EPS_予想_期末', 'EPS_予想_翌事業年度期末',
                '予想EPS', '発行済み株式数', '年度', '予測対象の年度の終了日']


        self.raw_cols = self._get_column_mapping(keys, raw_ccg.get_column_name)
        self.cols = self._get_column_mapping(keys, ccg.get_column_name)
        self.col_dtypes = self._get_column_mapping(keys, ccg.get_column_dtype)
        self.cols_plus_for_merger = self._get_column_mapping(keys, ccg.get_any_column_info, info_name = 'plus_for_merger')

    def _get_column_mapping(self, keys: list, column_config_getter: Callable, **kwargs):
        """
        指定されたキーリストに基づいて、ColumnConfigsGetter からカラム名のマッピング辞書を作成する。

        Parameters:
            keys (list): 取得するキーのリスト
            column_config_getter (ColumnConfigsGetterの関数): カラム名を取得用の関数

        Returns:
            dict: 指定されたキーと取得したカラム名の辞書
        """
        return {key: column_config_getter(key, **kwargs) for key in keys if column_config_getter(key, **kwargs) is not None}


    def _load_fin_data(self, raw_path: str) -> pd.DataFrame:
        """
        生の財務データを読み込み、空白値を NaN に変換する。

        Args:
            raw_path (str): 財務データのファイルパス

        Returns:
            pd.DataFrame: 読み込んだ財務データ
        """
        usecols = list(self.raw_cols.values())
        fin_df =  FileHandler.read_parquet(raw_path, usecols = usecols)
        return fin_df.replace('', np.nan)


    def _rename_columns(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        設定ファイルをもとにカラム名を変換する。

        Args:
            fin_df (pd.DataFrame): フィルタリング後の財務データ

        Returns:
            pd.DataFrame: カラム名変換後の財務データ
        """
    
        rename_mapping =  {self.raw_cols[key]: self.cols[key] for key in self.raw_cols.keys()}
        return fin_df.rename(columns=rename_mapping)

    def _format_dtypes(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        設定ファイルをもとに各カラムに適切なデータ型を設定。

        Args:
            fin_df (pd.DataFrame): カラム名変換後の財務データ

        Returns:
            pd.DataFrame: データ型変換後の財務データ
        """
        type_mapping = {self.cols[key]: eval(self.col_dtypes[key]) for key in self.raw_cols.keys()
                        if self.col_dtypes[key] != 'datetime'} 
        datetime_columns = [self.cols[key] for key in self.raw_cols.keys() if self.col_dtypes[key] == 'datetime']

        fin_df[[x for x in type_mapping.keys()]] = fin_df[[x for x in type_mapping.keys()]].astype(type_mapping)

        for column in datetime_columns:
            fin_df[column]= fin_df[column].astype(str).str[:10]
            fin_df[column] = pd.to_datetime(fin_df[column])
    
        fin_df[self.cols['銘柄コード']] = Formatter.format_stock_code(fin_df[self.cols['銘柄コード']])
        fin_df[self.cols['銘柄コード']] = fin_df[self.cols['銘柄コード']].replace(codes_to_replace_dict)
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
        fin_df = fin_df.loc[~fin_df[self.cols['開示書類種別']].isin(
            ['DividendForecastRevision', 'EarnForecastRevision', 'REITDividendForecastRevision', 'REITEarnForecastRevision'])]
        return fin_df.sort_values(self.cols['時刻']).drop_duplicates(subset=[self.cols['銘柄コード'], self.cols['日付']], keep="last") #コードと開示日が重複しているデータを、最後を残して削除

    def _calculate_additional_fins(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        追加の財務指標を計算。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        fin_df = self._merge_forecast_eps(fin_df)
        fin_df = self._calculate_outstanding_shares(fin_df)
        fin_df = self._append_fiscal_year_related_columns(fin_df)
        return fin_df


    def _merge_forecast_eps(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        fin_df.loc[
            (fin_df[self.cols['EPS_予想_期末']].notnull()) & (fin_df[self.cols['EPS_予想_翌事業年度期末']].notnull()), 
            self.cols['EPS_予想_翌事業年度期末']] = 0
        fin_df[[self.cols['EPS_予想_期末'], self.cols['EPS_予想_翌事業年度期末']]] = \
            fin_df[[self.cols['EPS_予想_期末'], self.cols['EPS_予想_翌事業年度期末']]].fillna(0)
        fin_df[self.cols['予想EPS']] = fin_df[self.cols['EPS_予想_期末']].values + fin_df[self.cols['EPS_予想_翌事業年度期末']].values
        return fin_df


    def _calculate_outstanding_shares(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        期末発行済株式数の算出。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        fin_df[self.cols['発行済み株式数']] = np.nan
        fin_df.loc[fin_df[self.cols['期末自己株式数']].notnull(), self.cols['発行済み株式数']] = \
            fin_df[self.cols['期末発行済株式数']] - fin_df[self.cols['期末自己株式数']]
        fin_df.loc[fin_df[self.cols['期末自己株式数']].isnull(), self.cols['発行済み株式数']] = \
            fin_df[self.cols['期末発行済株式数']]
        return fin_df
 

    def _append_fiscal_year_related_columns(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        年度に関する追加カラムを追加します。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        fin_df[self.cols['年度']] = fin_df[self.cols['当事業年度終了日']].dt.strftime("%Y")
        fin_df[self.cols['予測対象の年度の終了日']] = fin_df[self.cols['当事業年度終了日']].dt.strftime("%Y/%m")
        fin_df.loc[fin_df[self.cols['当会計期間の種類']] == "FY", self.cols['予測対象の年度の終了日']] = (
            fin_df.loc[fin_df[self.cols['当会計期間の種類']] == "FY", self.cols['当事業年度終了日']] \
            + pd.offsets.DateOffset(years=1)
                ).dt.strftime("%Y/%m")
        return fin_df
       

    def _process_merger(self, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """
        企業合併時の財務情報の合成を行います。

        Args:
            fin_df (pd.DataFrame): 重複データ削除後の財務データ

        Returns:
            pd.DataFrame: 追加計算後の財務データ
        """
        #合併前の各社のデータを足し合わせる項目
        plus_when_merging_mapping = {self.cols[k]: v for k, v in self.cols_plus_for_merger.items()}
        plus_when_merging = [k for k, v in plus_when_merging_mapping.items() if v]
        
        # 処理結果を格納するDataFrameを初期化
        result_df = stock_fin.copy()
        
        #合併リストの中身の分だけ処理を繰り返し
        for key, value in codes_to_merge_dict.items():
            # 現在の状態から合併する会社のデータを取得
            merger1 = result_df.loc[result_df[self.cols['銘柄コード']]==value['Code1']].sort_values(self.cols['日付']) #合併前1（存続）
            merger2 = result_df.loc[result_df[self.cols['銘柄コード']]==value['Code2']].sort_values(self.cols['日付']) #合併前2（消滅）
            
            #存続会社のデータを、合併前後に分ける。
            merger1_before = merger1[merger1[self.cols['日付']]<=value['MergerDate']]
            merger1_after = merger1[merger1[self.cols['日付']]>value['MergerDate']]
            
            #インデックスを揃える
            merger1_before = \
                pd.merge(merger1_before, merger2[self.cols['日付']], how='outer', on=self.cols['日付']).sort_values(self.cols['日付']).reset_index(drop=True)
            merger2 = \
                pd.merge(merger2, merger1_before[self.cols['日付']], how='outer', on=self.cols['日付']).sort_values(self.cols['日付']).reset_index(drop=True)
            
            #NaN値を埋める
            merger1_before = merger1_before.ffill()
            merger1_before = merger1_before.bfill()
            merger2 = merger2.ffill()
            merger2 = merger2.bfill()
            
            #merger1_beforeの値を書き換えていく。
            merger1_before[self.cols['銘柄コード']] = key
            merger1_before[self.cols['発行済み株式数']] = merger1_before[self.cols['発行済み株式数']] + \
                                                merger2[self.cols['発行済み株式数']] * value['ExchangeRate']
            merger1_before[plus_when_merging] =  merger1_before[plus_when_merging] + merger2[plus_when_merging]
            
            #合併前後のデータを結合する。
            merged = pd.concat([merger1_before, merger1_after], axis=0).sort_values(self.cols['日付'])
            
            # 処理済みのデータをresult_dfから削除
            result_df = result_df[(result_df[self.cols['銘柄コード']]!=value['Code1'])&(result_df[self.cols['銘柄コード']]!=value['Code2'])]
            
            # 新しい合併データを追加
            result_df = pd.concat([result_df, merged], axis=0)
        
        # 全ての処理が終わってからソートして返す
        return result_df.sort_values([self.cols['日付'], self.cols['銘柄コード']])

    def _finalize_df(self, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """
        最終データの整形処理を行う。

        Args:
            stock_fin (pd.DataFrame): 財務データの最終処理前の状態

        Returns:
            pd.DataFrame: 最終処理後のデータ
        """
        return stock_fin.drop([self.cols['時刻']], axis=1).drop_duplicates(keep='last').reset_index(drop=True) # データフレームの最終処理


if __name__ == '__main__':
    FinProcessor()