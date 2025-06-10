from __future__ import annotations

from typing import Dict, Optional, Literal
import pandas as pd

from acquisition.jquants_api_operations import StockAcquisitionFacade
from acquisition.features_updater.facades import FeaturesUpdateFacade


class DataUpdateFacade:
    """Facade for updating and loading data."""

    def __init__(self, mode: Literal['update_and_load', 'load_only', 'none'], universe_filter: str) -> None:
        self.mode = mode
        self.universe_filter = universe_filter

    async def execute(self) -> Optional[Dict[str, pd.DataFrame]]:
        if self.mode == 'none':
            return None
        update = self.mode == 'update_and_load'
        process = update
        stock_dict = StockAcquisitionFacade(update=update, process=process, filter=self.universe_filter).get_stock_data_dict()
        if update:
            fu = FeaturesUpdateFacade()
            await fu.update_all()
        return stock_dict
