import pandas as pd
from trading.sbi.selection.interface import ISectorProvider

class SectorProvider(ISectorProvider):
    """セクター情報提供クラス"""
    
    def __init__(self, sector_definitions_csv: str):
        """
        Args:
            sector_definitions_csv: セクター定義CSVファイルのパス
        """
        self.sector_definitions_csv = sector_definitions_csv
    
    def get_sector_definitions(self) -> pd.DataFrame:
        """セクター定義情報を取得"""
        df = pd.read_csv(self.sector_definitions_csv)
        df['Code'] = df['Code'].astype(str)
        df = df[df['Current']==1] #TODO ここの動作確認が必要
        return df