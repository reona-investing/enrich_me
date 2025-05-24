import pandas as pd
import numpy as np
from datetime import datetime
from acquisition.refactored_jqapi.processors.base_processor import BaseProcessor
from acquisition.refactored_jqapi.schemas import RawStockFinSchema, StockFinSchema
from acquisition.refactored_jqapi.utils import DataFormatter
from acquisition.refactored_jqapi.config import MergerConfig

class FinProcessor(BaseProcessor):
    """財務データ処理クラス（RawSchemaを使用）"""
    
    def __init__(self, raw_path: str, processed_path: str):
        super().__init__(raw_path, processed_path)
        self.raw_schema = RawStockFinSchema
        self.schema = StockFinSchema
    
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """財務データの処理"""
        print(f"Starting financial data processing with {len(data)} records")
        print(f"Available columns: {list(data.columns)}")
        
        # 空白値をNaNに変換
        data = data.replace('', np.nan)
        
        # 生データスキーマで利用可能なkeyを特定
        available_keys = self.raw_schema.get_available_keys(data)
        print(f"Available keys (from raw schema): {available_keys}")
        
        # 生データスキーマでカラム名をリネーム（raw_name -> processed_name）
        rename_mapping = self.raw_schema.get_available_rename_mapping(data)
        print(f"Rename mapping (raw -> processed): {rename_mapping}")
        data = data.rename(columns=rename_mapping)
        
        # データ型変換
        data = self._format_dtypes(data, available_keys)
        
        # 重複データの削除
        data = self._drop_duplicated_data(data)
        
        # 追加財務指標の計算
        data = self._calculate_additional_fins(data)
        
        # 企業合併処理
        data = self._process_merger(data)
        
        # 最終整形
        data = self._finalize_data(data)
        
        # 最終スキーマで検証（利用可能なカラムのみ）
        data = self.schema.validate_dataframe(data)
        
        print(f"Financial data processing completed with {len(data)} records")
        print(f"Final columns: {list(data.columns)}")
        return data
    
    def _format_dtypes(self, data: pd.DataFrame, available_keys: list) -> pd.DataFrame:
        """データ型の変換（利用可能なキーのみ）"""
        data = data.copy()
        raw_columns = self.raw_schema.get_columns()
        
        # 日付カラムの変換
        for key in available_keys:
            col_def = raw_columns[key]
            processed_name = col_def.processed_name
            
            if processed_name in data.columns:
                # 最終スキーマでの期待データ型をチェック
                final_columns = self.schema.get_columns()
                if key in final_columns:
                    expected_dtype = final_columns[key].dtype
                    
                    try:
                        if expected_dtype == datetime:
                            data[processed_name] = DataFormatter.convert_to_datetime(data[processed_name])
                            print(f"Converted date column: {processed_name}")
                        elif expected_dtype in [float, int]:
                            data[processed_name] = pd.to_numeric(data[processed_name], errors='coerce')
                            print(f"Converted numeric column: {processed_name}")
                        else:
                            data[processed_name] = data[processed_name].astype(str)
                            print(f"Converted string column: {processed_name}")
                    except Exception as e:
                        print(f"データ型変換エラー {processed_name}: {e}")
        
        # 銘柄コードの特別処理
        if 'LocalCode' in data.columns:
            try:
                data['LocalCode'] = DataFormatter.format_stock_code(data['LocalCode'].astype(str))
                data['LocalCode'] = DataFormatter.replace_stock_codes(data['LocalCode'])
                print("Processed stock code column")
            except Exception as e:
                print(f"銘柄コード変換エラー: {e}")
        
        return data
    
    def _drop_duplicated_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """重複データの削除"""
        initial_count = len(data)
        
        # 特定の開示書類種別を除外
        if 'TypeOfDocument' in data.columns:
            exclude_types = [
                'DividendForecastRevision', 'EarnForecastRevision',
                'REITDividendForecastRevision', 'REITEarnForecastRevision'
            ]
            data = data[~data['TypeOfDocument'].isin(exclude_types)]
            print(f"Excluded revision documents: {initial_count - len(data)} records")
        
        # 同日の重複を最新時刻で解決
        duplicate_cols = []
        if 'LocalCode' in data.columns:
            duplicate_cols.append('LocalCode')
        if 'DisclosedDate' in data.columns:
            duplicate_cols.append('DisclosedDate')
        
        if len(duplicate_cols) >= 2:
            before_dedup = len(data)
            if 'DisclosedTime' in data.columns:
                data = data.sort_values('DisclosedTime')
            data = data.drop_duplicates(subset=duplicate_cols, keep="last")
            print(f"Removed duplicates: {before_dedup - len(data)} records")
        
        return data
    
    def _calculate_additional_fins(self, data: pd.DataFrame) -> pd.DataFrame:
        """追加財務指標の計算"""
        print("Calculating additional financial metrics...")
        data = self._merge_forecast_eps(data)
        data = self._calculate_outstanding_shares(data)
        data = self._append_fiscal_year_columns(data)
        return data
    
    def _merge_forecast_eps(self, data: pd.DataFrame) -> pd.DataFrame:
        """予想EPSの統合"""
        print("Merging forecast EPS...")
        
        # 予想EPSカラムのデフォルト値設定
        if 'ForecastEarningsPerShare' not in data.columns:
            data['ForecastEarningsPerShare'] = 0.0
        if 'NextYearForecastEarningsPerShare' not in data.columns:
            data['NextYearForecastEarningsPerShare'] = 0.0
        
        # 数値型に変換
        data['ForecastEarningsPerShare'] = pd.to_numeric(
            data['ForecastEarningsPerShare'], errors='coerce'
        ).fillna(0.0)
        data['NextYearForecastEarningsPerShare'] = pd.to_numeric(
            data['NextYearForecastEarningsPerShare'], errors='coerce'
        ).fillna(0.0)
        
        # 両方の予想EPSがある場合の処理
        both_exist = (
            (data['ForecastEarningsPerShare'] != 0) & 
            (data['NextYearForecastEarningsPerShare'] != 0)
        )
        if both_exist.any():
            data.loc[both_exist, 'NextYearForecastEarningsPerShare'] = 0.0
            print(f"Reset NextYear EPS for {both_exist.sum()} records with both forecasts")
        
        # 統合予想EPSを計算
        data['ForecastEPS'] = (
            data['ForecastEarningsPerShare'] + 
            data['NextYearForecastEarningsPerShare']
        )
        
        print(f"Successfully calculated ForecastEPS for {len(data)} records")
        return data
    
    def _calculate_outstanding_shares(self, data: pd.DataFrame) -> pd.DataFrame:
        """発行済み株式数の計算"""
        print("Calculating outstanding shares...")
        
        issued_col = 'NumberOfIssuedAndOutstandingSharesAtTheEndOfFiscalYearIncludingTreasuryStock'
        treasury_col = 'NumberOfTreasuryStockAtTheEndOfFiscalYear'
        
        data['OutstandingShares'] = np.nan
        
        if issued_col in data.columns:
            data[issued_col] = pd.to_numeric(data[issued_col], errors='coerce')
            
            if treasury_col in data.columns:
                data[treasury_col] = pd.to_numeric(data[treasury_col], errors='coerce')
                
                # 自己株式数がある場合
                has_treasury = data[treasury_col].notnull()
                data.loc[has_treasury, 'OutstandingShares'] = (
                    data.loc[has_treasury, issued_col] - data.loc[has_treasury, treasury_col]
                )
                
                # 自己株式数がない場合
                no_treasury = data[treasury_col].isnull()
                data.loc[no_treasury, 'OutstandingShares'] = data.loc[no_treasury, issued_col]
                
                print(f"Calculated outstanding shares: {has_treasury.sum()} with treasury, {no_treasury.sum()} without")
            else:
                data['OutstandingShares'] = data[issued_col]
                print("Used issued shares as outstanding shares (no treasury data)")
        
        return data
    
    def _append_fiscal_year_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """年度関連カラムの追加"""
        print("Adding fiscal year columns...")
        
        if 'CurrentFiscalYearEndDate' in data.columns:
            try:
                # 決算年度
                data['CurrentFiscalYear'] = data['CurrentFiscalYearEndDate'].dt.year
                
                # 予測対象年度終了日
                data['ForecastFiscalYearEndDate'] = data['CurrentFiscalYearEndDate'].dt.strftime("%Y/%m")
                
                # FY（通期）の場合は翌年度とする
                if 'TypeOfCurrentPeriod' in data.columns:
                    fy_condition = data['TypeOfCurrentPeriod'] == "FY"
                    if fy_condition.any():
                        next_year_dates = (
                            data.loc[fy_condition, 'CurrentFiscalYearEndDate'] + 
                            pd.offsets.DateOffset(years=1)
                        )
                        data.loc[fy_condition, 'ForecastFiscalYearEndDate'] = next_year_dates.dt.strftime("%Y/%m")
                        print(f"Adjusted forecast year for {fy_condition.sum()} FY records")
                
                print("Successfully added fiscal year columns")
            except Exception as e:
                print(f"年度カラム追加エラー: {e}")
                data['CurrentFiscalYear'] = 2024
                data['ForecastFiscalYearEndDate'] = "2024/03"
        
        return data
    
    def _process_merger(self, data: pd.DataFrame) -> pd.DataFrame:
        """企業合併処理"""
        # 処理済みカラム名で判定
        if 'LocalCode' not in data.columns or 'DisclosedDate' not in data.columns:
            print("Skipping merger processing: required columns not available")
            return data
        
        print("Processing mergers...")
        
        try:
            result_df = data.copy()
            additive_cols = self.schema.get_merger_additive_columns()
            print(f"Merger additive columns: {additive_cols}")
            
            for key, merger_info in MergerConfig.MERGER_INFO.items():
                print(f"Processing merger: {key}")
                result_df = self._process_single_merger(result_df, key, merger_info, additive_cols)
            
            return result_df.sort_values(['DisclosedDate', 'LocalCode'])
        except Exception as e:
            print(f"合併処理エラー: {e}")
            return data
    
    def _process_single_merger(self, data: pd.DataFrame, merged_code: str, 
                             merger_info: dict, additive_cols: list) -> pd.DataFrame:
        """個別の合併処理"""
        try:
            company1_data = data[data['LocalCode'] == merger_info['Code1']].sort_values('DisclosedDate')
            company2_data = data[data['LocalCode'] == merger_info['Code2']].sort_values('DisclosedDate')
            
            if company1_data.empty or company2_data.empty:
                return data
            
            merger_date = merger_info['MergerDate']
            company1_before = company1_data[company1_data['DisclosedDate'] <= merger_date]
            company1_after = company1_data[company1_data['DisclosedDate'] > merger_date]
            
            merged_before = self._merge_financial_data(
                company1_before, company2_data, merger_info, merged_code, additive_cols
            )
            
            merged_data = pd.concat([merged_before, company1_after], axis=0).sort_values('DisclosedDate')
            
            data = data[
                (data['LocalCode'] != merger_info['Code1']) & 
                (data['LocalCode'] != merger_info['Code2'])
            ]
            data = pd.concat([data, merged_data], axis=0)
            
            return data
        except Exception as e:
            print(f"個別合併処理エラー {merged_code}: {e}")
            return data
    
    def _merge_financial_data(self, company1_data: pd.DataFrame, company2_data: pd.DataFrame, 
                            merger_info: dict, merged_code: str, additive_cols: list) -> pd.DataFrame:
        """財務データの合併"""
        try:
            # 日付でインデックスを揃える
            company1_aligned = pd.merge(
                company1_data, company2_data[['DisclosedDate']], 
                how='outer', on='DisclosedDate'
            ).sort_values('DisclosedDate').ffill().bfill()
            
            company2_aligned = pd.merge(
                company2_data, company1_data[['DisclosedDate']], 
                how='outer', on='DisclosedDate'
            ).sort_values('DisclosedDate').ffill().bfill()
            
            # 合併データの作成
            merged_data = company1_aligned.copy()
            merged_data['LocalCode'] = merged_code
            
            # 発行済み株式数の合併
            if 'OutstandingShares' in merged_data.columns:
                company1_shares = pd.to_numeric(company1_aligned['OutstandingShares'], errors='coerce').fillna(0)
                company2_shares = pd.to_numeric(company2_aligned['OutstandingShares'], errors='coerce').fillna(0)
                merged_data['OutstandingShares'] = (
                    company1_shares + company2_shares * merger_info['ExchangeRate']
                )
            
            # 加算対象項目の合併
            available_additive_cols = [col for col in additive_cols if col in merged_data.columns]
            for col in available_additive_cols:
                company1_values = pd.to_numeric(company1_aligned[col], errors='coerce').fillna(0)
                company2_values = pd.to_numeric(company2_aligned[col], errors='coerce').fillna(0)
                merged_data[col] = company1_values + company2_values
            
            return merged_data
        except Exception as e:
            print(f"財務データ合併エラー: {e}")
            return company1_data
    
    def _finalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """最終データ整形"""
        print("Finalizing data...")
        
        initial_count = len(data)
        
        # 不要なカラムを削除
        if 'DisclosedTime' in data.columns:
            data = data.drop('DisclosedTime', axis=1)
        
        # 重複削除
        data = data.drop_duplicates(keep='last').reset_index(drop=True)
        removed_count = initial_count - len(data)
        
        if removed_count > 0:
            print(f"Final cleanup: removed {removed_count} duplicate records")
        
        return data


if __name__ == '__main__':
    from utils.paths import Paths
    raw = Paths.RAW_STOCK_FIN_PARQUET
    processed = Paths.STOCK_FIN_PARQUET
    fp = FinProcessor(raw, processed)
    data = fp.process()
    print(data.head(2))