import pandas as pd
from datetime import datetime
from utils.jquants_api_utils import cli
from utils.flag_manager import flag_manager, Flags
from acquisition.refactored_jqapi.utils import FileHandler

class PriceUpdater:
    """株価データ更新クラス"""
    
    def __init__(self, basic_path: str):
        self.basic_path = basic_path
        self.current_year = datetime.today().year
        self.prev_year = self.current_year - 1
        
    def update(self) -> pd.DataFrame:
        """年次ファイル形式での更新"""
        current_path = self._get_yearly_path(self.current_year)
        prev_path = self._get_yearly_path(self.prev_year)
        
        # 今年のファイルが存在しない場合、前年を更新
        if not FileHandler.file_exists(current_path):
            self._update_yearly_file(self.prev_year, prev_path)
        
        # 今年のファイルを更新
        return self._update_yearly_file(self.current_year, current_path)
    
    def _get_yearly_path(self, year: int) -> str:
        """年次パスの生成"""
        return self.basic_path.replace('0000', str(year))
    
    def _update_yearly_file(self, year: int, path: str) -> pd.DataFrame:
        """年次ファイルの更新"""
        existing_data = self._load_existing_data(path) if FileHandler.file_exists(path) else pd.DataFrame()
        new_data = self._fetch_yearly_data(year, existing_data)
        merged_data = self._merge_data(existing_data, new_data)
        formatted_data = self._format_data(merged_data)
        
        FileHandler.write_parquet(formatted_data, path)
        return formatted_data
    
    def _load_existing_data(self, path: str) -> pd.DataFrame:
        """既存データの読み込み"""
        return FileHandler.read_parquet(path)
    
    def _fetch_yearly_data(self, year: int, existing_data: pd.DataFrame) -> pd.DataFrame:
        """年次データの取得"""
        if existing_data.empty:
            return self._fetch_full_year_data(year)
        
        # 日付カラムを特定
        date_columns = ['Date', '日付']
        date_col = None
        
        for col in date_columns:
            if col in existing_data.columns:
                date_col = col
                break
        
        if date_col is None:
            return self._fetch_full_year_data(year)
        
        # 最終日付から今日まで取得
        try:
            existing_data[date_col] = pd.to_datetime(existing_data[date_col], format='mixed', errors='coerce')
            last_date = existing_data[date_col].max().date()
            
            if last_date != datetime.today().date():
                new_data = self._fetch_partial_data(last_date)
                self._check_adjustment_flag(new_data)
                return new_data
        except Exception as e:
            print(f"日付処理エラー: {e}. 全期間データを取得します。")
            return self._fetch_full_year_data(year)
        
        return pd.DataFrame()
    
    def _fetch_full_year_data(self, year: int) -> pd.DataFrame:
        """年間全データの取得"""
        return cli.get_price_range(
            start_dt=datetime(year, 1, 1),
            end_dt=datetime(year, 12, 31)
        )
    
    def _fetch_partial_data(self, start_date: datetime) -> pd.DataFrame:
        """部分データの取得"""
        return cli.get_price_range(start_dt=start_date, end_dt=datetime.today())
    
    def _check_adjustment_flag(self, data: pd.DataFrame) -> None:
        """調整係数変更フラグの確認"""
        adjustment_columns = ['AdjustmentFactor', '調整係数']
        
        for col in adjustment_columns:
            if col in data.columns and any(data[col] != 1):
                flag_manager.set_flag(Flags.PROCESS_STOCK_PRICE, True)
                break
    
    def _merge_data(self, existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
        """データのマージ"""
        if existing_data.empty:
            return new_data
        
        combined = pd.concat([existing_data, new_data], axis=0)
        
        # 日付とコードカラムを特定
        date_columns = ['Date', '日付']
        code_columns = ['Code', '銘柄コード']
        
        date_col = None
        code_col = None
        
        for col in date_columns:
            if col in combined.columns:
                date_col = col
                break
        
        for col in code_columns:
            if col in combined.columns:
                code_col = col
                break
        
        if date_col and code_col:
            combined[date_col] = pd.to_datetime(combined[date_col], format='mixed', errors='coerce')
            return combined.drop_duplicates(subset=[date_col, code_col], keep='last')
        
        return combined.drop_duplicates()
    
    def _format_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """データの整形"""
        return data.dropna().reset_index(drop=True)


if __name__ == '__main__':
    from utils.paths import Paths
    path = Paths.RAW_STOCK_PRICE_PARQUET
    pu = PriceUpdater(path)
    df = pu.update()
    print(df.tail(2))