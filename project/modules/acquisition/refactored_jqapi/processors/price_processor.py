import pandas as pd
from datetime import datetime
from typing import Tuple
from acquisition.refactored_jqapi.schemas import RawStockPriceSchema, StockPriceSchema
from acquisition.refactored_jqapi.utils import DataFormatter
from acquisition.refactored_jqapi.config import MergerConfig
from utils.flag_manager import flag_manager, Flags
from acquisition.refactored_jqapi.utils import FileHandler

class PriceProcessor:
    """株価データ処理クラス（RawSchemaを使用）"""
    
    def __init__(self, raw_path: str, processed_path: str):
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.raw_schema = RawStockPriceSchema
        self.schema = StockPriceSchema
        self.temp_cumprod = {}
    
    def process(self) -> pd.DataFrame:
        """年次ファイルでの処理"""
        print(f"Processing price data with pattern: {self.raw_path}")
        
        end_date = datetime.today()
        temp_cumprod = {}
        
        for year in range(end_date.year, 2012, -1):
            is_latest_file = year == end_date.year
            should_process = is_latest_file or flag_manager.get_flag(Flags.PROCESS_STOCK_PRICE)
            
            if should_process:
                yearly_raw_path = self.raw_path.replace('0000', str(year))
                yearly_processed_path = self.processed_path.replace('0000', str(year))
                
                try:
                    stock_price = self._load_yearly_raw_data(yearly_raw_path)
                    if stock_price.empty:
                        print(f"No data found for year {year}")
                        continue
                    
                    print(f"Processing {len(stock_price)} records for year {year}")
                    stock_price, temp_cumprod = self._process_stock_price(stock_price, temp_cumprod, is_latest_file)
                    self._save_yearly_data(stock_price, yearly_processed_path)
                    
                except Exception as e:
                    print(f"年次処理エラー {year}: {e}")
        
        # 最新年のデータを返す
        latest_path = self.processed_path.replace('0000', str(end_date.year))
        try:
            return FileHandler.read_parquet(latest_path)
        except:
            return pd.DataFrame()
    
    def _load_yearly_raw_data(self, yearly_path: str) -> pd.DataFrame:
        """年次生データの読み込み"""
        try:
            if not FileHandler.file_exists(yearly_path):
                return pd.DataFrame()
            
            # ファイル全体を読み込み
            df = FileHandler.read_parquet(yearly_path)
            print(f"Loaded raw data columns: {list(df.columns)}")
            
            # 生データスキーマで利用可能なkeyを特定
            available_keys = self.raw_schema.get_available_keys(df)
            print(f"Available raw price keys: {available_keys}")
            
            # 生データスキーマでカラム名をリネーム
            rename_mapping = self.raw_schema.get_available_rename_mapping(df)
            print(f"Price rename mapping: {rename_mapping}")
            df = df.rename(columns=rename_mapping)
            
            return df
            
        except Exception as e:
            print(f"年次データ読み込みエラー {yearly_path}: {e}")
            return pd.DataFrame()
    
    def _process_stock_price(self, stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool) -> Tuple[pd.DataFrame, dict]:
        """株価データの処理"""
        try:
            print(f"Processing columns: {list(stock_price.columns)}")
            
            # 銘柄コードのフォーマット
            if 'Code' in stock_price.columns:
                stock_price['Code'] = DataFormatter.format_stock_code(stock_price['Code'].astype(str))
                stock_price['Code'] = DataFormatter.replace_stock_codes(stock_price['Code'])
            
            # 売買停止期間の補完
            stock_price = self._fill_suspension_period(stock_price)
            
            # データ型変換
            stock_price = self._format_dtypes(stock_price)
            
            # システム障害日の除外
            stock_price = self._remove_system_failure_day(stock_price)
            
            # 累積調整係数の適用
            stock_price, temp_cumprod = self._apply_cumulative_adjustment_factor(
                stock_price, temp_cumprod, is_latest_file
            )
            
            # 最終整形
            stock_price = self._finalize_price_data(stock_price)
            
            # スキーマ検証
            stock_price = self.schema.validate_dataframe(stock_price)
            
            return stock_price, temp_cumprod
            
        except Exception as e:
            print(f"株価処理エラー: {e}")
            return stock_price, temp_cumprod
    
    def _fill_suspension_period(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """売買停止期間のデータ補完"""
        try:
            if 'Date' not in stock_price.columns or 'Code' not in stock_price.columns:
                return stock_price
            
            rows_to_add = []
            date_list = stock_price['Date'].unique()
            codes_after_replacement = MergerConfig.CODE_REPLACEMENT.values()
            
            for code_replaced in codes_after_replacement:
                dates_to_fill = self._get_missing_dates(stock_price, code_replaced, date_list)
                rows_to_add.extend(self._create_missing_rows(stock_price, code_replaced, dates_to_fill))
            
            if rows_to_add:
                return pd.concat([stock_price] + rows_to_add, axis=0, ignore_index=True)
            return stock_price
            
        except Exception as e:
            print(f"売買停止期間補完エラー: {e}")
            return stock_price
    
    def _get_missing_dates(self, stock_price: pd.DataFrame, code: str, date_list) -> list:
        """欠損している日付を取得"""
        existing_dates = stock_price[stock_price['Code'] == code]['Date'].unique()
        return [d for d in date_list if d not in existing_dates]
    
    def _create_missing_rows(self, stock_price: pd.DataFrame, code: str, dates_to_fill: list) -> list:
        """欠損期間の行を作成"""
        rows = []
        if len(dates_to_fill) <= 5:  # 短期間の欠損のみ補完
            for date in dates_to_fill:
                prev_data = stock_price[
                    (stock_price['Code'] == code) & (stock_price['Date'] <= date)
                ].sort_values('Date')
                
                if not prev_data.empty:
                    last_close = prev_data.iloc[-1]['Close'] if 'Close' in prev_data.columns else 100
                    row = {
                        'Date': date,
                        'Code': code,
                        'Open': last_close,
                        'High': last_close,
                        'Low': last_close,
                        'Close': last_close,
                        'Volume': 0,
                        'TurnoverValue': 0,
                        'AdjustmentFactor': 1
                    }
                    # 利用可能なカラムのみ追加
                    available_row = {k: v for k, v in row.items() if k in stock_price.columns}
                    rows.append(pd.DataFrame([available_row]))
        return rows
    
    def _format_dtypes(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """データ型の変換"""
        try:
            stock_price = stock_price.copy()
            
            # 日付の変換
            if 'Date' in stock_price.columns:
                stock_price['Date'] = DataFormatter.convert_to_datetime(stock_price['Date'])
            
            # 数値カラムの変換
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 
                           'TurnoverValue', 'AdjustmentFactor']
            available_numeric_cols = [col for col in numeric_cols if col in stock_price.columns]
            
            for col in available_numeric_cols:
                stock_price[col] = pd.to_numeric(stock_price[col], errors='coerce')
            
            return stock_price
            
        except Exception as e:
            print(f"データ型変換エラー: {e}")
            return stock_price
    
    def _remove_system_failure_day(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """システム障害日の除外"""
        if 'Date' in stock_price.columns:
            return stock_price[stock_price['Date'] != '2020-10-01']
        return stock_price
    
    def _apply_cumulative_adjustment_factor(self, stock_price: pd.DataFrame, temp_cumprod: dict, is_latest_file: bool) -> Tuple[pd.DataFrame, dict]:
        """累積調整係数の適用"""
        try:
            if 'Code' not in stock_price.columns or 'Date' not in stock_price.columns:
                return stock_price, temp_cumprod
            
            stock_price = stock_price.sort_values(['Code', 'Date']).copy()
            
            # 累積調整係数の計算
            stock_price = stock_price.groupby('Code', group_keys=False).apply(
                self._calculate_cumulative_adjustment_factor
            ).reset_index(drop=True)
            
            if not is_latest_file and temp_cumprod:
                stock_price = self._inherit_cumulative_values(stock_price, temp_cumprod)
            
            # 各銘柄の最初の日の累積調整係数を保存（遡って処理するため）
            new_temp_cumprod = stock_price.groupby('Code')['CumulativeAdjustmentFactor'].first().to_dict()
            temp_cumprod.update(new_temp_cumprod)
            
            # 手動調整の適用
            stock_price = self._apply_manual_adjustments(stock_price)
            
            # 価格・出来高への調整係数適用
            price_volume_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_price_cols = [col for col in price_volume_cols if col in stock_price.columns]
            
            for col in available_price_cols:
                stock_price[col] = stock_price[col] / stock_price['CumulativeAdjustmentFactor']
            
            return stock_price, temp_cumprod
            
        except Exception as e:
            print(f"累積調整係数適用エラー: {e}")
            return stock_price, temp_cumprod
    
    def _calculate_cumulative_adjustment_factor(self, group_data: pd.DataFrame) -> pd.DataFrame:
        """累積調整係数の計算"""
        try:
            group_data = group_data.sort_values('Date', ascending=False).copy()
            
            if 'AdjustmentFactor' in group_data.columns:
                group_data['AdjustmentFactor'] = group_data['AdjustmentFactor'].shift(1).fillna(1.0)
                group_data['CumulativeAdjustmentFactor'] = 1 / group_data['AdjustmentFactor'].cumprod()
            else:
                group_data['CumulativeAdjustmentFactor'] = 1.0
            
            return group_data.sort_values('Date')
            
        except Exception as e:
            print(f"累積調整係数計算エラー: {e}")
            group_data['CumulativeAdjustmentFactor'] = 1.0
            return group_data
    
    def _inherit_cumulative_values(self, stock_price: pd.DataFrame, temp_cumprod: dict) -> pd.DataFrame:
        """計算途中の暫定累積調整係数を引き継ぎ"""
        try:
            stock_price['InheritedValue'] = stock_price['Code'].map(temp_cumprod).fillna(1)
            stock_price['CumulativeAdjustmentFactor'] *= stock_price['InheritedValue']
            return stock_price.drop(columns='InheritedValue')
        except Exception as e:
            print(f"累積値継承エラー: {e}")
            return stock_price
    
    def _apply_manual_adjustments(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """手動調整の適用"""
        try:
            for adjustment in MergerConfig.MANUAL_ADJUSTMENTS:
                condition = (
                    (stock_price['Code'] == adjustment['Code']) & 
                    (stock_price['Date'] < adjustment['Date'])
                )
                stock_price.loc[condition, 'CumulativeAdjustmentFactor'] *= adjustment['Rate']
            return stock_price
        except Exception as e:
            print(f"手動調整エラー: {e}")
            return stock_price
    
    def _finalize_price_data(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """最終データ整形"""
        try:
            if 'Code' in stock_price.columns:
                stock_price = stock_price.dropna(subset=['Code'])
            
            duplicate_cols = [col for col in ['Date', 'Code'] if col in stock_price.columns]
            if len(duplicate_cols) >= 2:
                stock_price = stock_price.drop_duplicates(subset=duplicate_cols, keep='last')
            
            if 'Date' in stock_price.columns and 'Code' in stock_price.columns:
                stock_price = stock_price.sort_values(['Date', 'Code'])
            
            return stock_price.reset_index(drop=True)
            
        except Exception as e:
            print(f"最終整形エラー: {e}")
            return stock_price
    
    def _save_yearly_data(self, df: pd.DataFrame, yearly_processed_path: str) -> None:
        """年次の加工後価格データを保存"""
        FileHandler.write_parquet(df, yearly_processed_path)

if __name__ == '__main__':
    from utils.paths import Paths
    raw = Paths.RAW_STOCK_PRICE_PARQUET
    processed = Paths.STOCK_PRICE_PARQUET
    pp = PriceProcessor(raw, processed)
    data = pp.process()
    print(data.sort_values('CumulativeAdjustmentFactor').tail(2))
    print(data.sort_values('CumulativeAdjustmentFactor+69').head(2))