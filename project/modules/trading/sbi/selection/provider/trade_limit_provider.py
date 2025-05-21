import pandas as pd
from trading.sbi import TradePossibilityManager
from trading.sbi.common.interface import IMarginProvider
from trading.sbi.selection.interface import ITradeLimitProvider

class TradeLimitProvider(ITradeLimitProvider):
    """取引制限情報提供クラス"""
    
    def __init__(self, trade_possibility_manager: TradePossibilityManager, margin_provider: IMarginProvider):
        self.trade_possibility_manager = trade_possibility_manager
        self.margin_provider = margin_provider
        self._cache = {
            'buyable_symbols': None,
            'sellable_symbols': None,
            'margin_power': None,
            'last_fetch_time': None
        }
        self._fetched = False
    
    async def _ensure_fetched(self):
        """データが取得済みであることを確認"""
        if not self._fetched:
            print("[INFO] TradePossibilityManager.fetch() を実行します")
            await self.trade_possibility_manager.fetch()
            await self.margin_provider.refresh()
            self._fetched = True
            self._cache['last_fetch_time'] = pd.Timestamp.now()
    
    async def get_buyable_symbols(self) -> pd.DataFrame:
        """買い可能な銘柄リストを取得"""
        await self._ensure_fetched()
        
        if self._cache['buyable_symbols'] is None:
            self._cache['buyable_symbols'] = pd.DataFrame({
                'Code': list(self.trade_possibility_manager.data_dict['buyable_limits'].keys()),
                'MaxUnit': list(self.trade_possibility_manager.data_dict['buyable_limits'].values())
            })
        
        return self._cache['buyable_symbols'].copy()
    
    async def get_sellable_symbols(self) -> pd.DataFrame:
        """売り可能な銘柄リストを取得"""
        await self._ensure_fetched()
        
        if self._cache['sellable_symbols'] is None:
            sellable_codes = list(self.trade_possibility_manager.data_dict['sellable_limits'].keys())
            self._cache['sellable_symbols'] = pd.DataFrame({
                'Code': sellable_codes,
                'MaxUnit': [self.trade_possibility_manager.data_dict['sellable_limits'][code] for code in sellable_codes],
                'isBorrowingStock': [self.trade_possibility_manager.data_dict['borrowing_stocks'][code] for code in sellable_codes]
            })
        
        return self._cache['sellable_symbols'].copy()
    
    async def get_margin_power(self) -> float:
        """信用建余力を取得"""
        await self._ensure_fetched()
        
        if self._cache['margin_power'] is None:
            self._cache['margin_power'] = await self.margin_provider.get_available_margin()
        
        return self._cache['margin_power']
    
    async def refresh(self):
        """キャッシュを強制的に更新"""
        self._fetched = False
        self._cache = {
            'buyable_symbols': None,
            'sellable_symbols': None,
            'margin_power': None,
            'last_fetch_time': None
        }
        await self._ensure_fetched()