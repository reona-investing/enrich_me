from abc import ABC, abstractmethod
import pandas as pd


class ISectorProvider(ABC):
    """セクター情報提供インターフェース"""
    @abstractmethod
    def get_sector_definitions(self) -> pd.DataFrame:
        """セクター定義情報を取得"""
        pass

class IPriceProvider(ABC):
    """価格情報提供インターフェース"""
    @abstractmethod
    def get_price_data(self) -> pd.DataFrame:
        """価格データを取得"""
        pass
    
    @abstractmethod
    def get_etf_price(self, symbol_code: str) -> float:
        """指定したETFの価格を取得"""
        pass

class ITradeLimitProvider(ABC):
    """取引制限情報提供インターフェース"""
    @abstractmethod
    async def get_buyable_symbols(self) -> pd.DataFrame:
        """買い可能な銘柄リストを取得"""
        pass
    
    @abstractmethod
    async def get_sellable_symbols(self) -> pd.DataFrame:
        """売り可能な銘柄リストを取得"""
        pass
    
    @abstractmethod
    async def get_margin_power(self) -> float:
        """信用建余力を取得"""
        pass

