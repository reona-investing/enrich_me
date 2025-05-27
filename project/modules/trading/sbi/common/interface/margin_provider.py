from abc import ABC, abstractmethod

class IMarginProvider(ABC):
    """証拠金情報提供のインターフェース"""
    
    @abstractmethod
    async def get_available_margin(self) -> float:
        """利用可能な証拠金を取得する"""
        pass
    
    @abstractmethod
    async def refresh(self) -> None:
        """証拠金情報を更新する"""
        pass