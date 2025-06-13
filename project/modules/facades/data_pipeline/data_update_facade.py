from __future__ import annotations

from typing import Dict, List, Optional, Literal
import pandas as pd

from acquisition.jquants_api_operations import StockAcquisitionFacade
from acquisition.features_updater.facades import FeaturesUpdateFacade


class DataUpdateFacade:
    """データの更新及び読み込みを担当するファサード"""

    def __init__(self, 
                 mode: Literal['update_and_load', 'load_only', 'none'], 
                 universe_filter: str,
                 filtered_code_list: List[str] | None = None) -> None:
        self.mode = mode
        self.universe_filter = universe_filter
        self.filtered_code_list = filtered_code_list

    async def execute(self) -> Optional[Dict[str, pd.DataFrame]]:
        if self.mode == 'none':
            return None
        update = self.mode == 'update_and_load'
        process = update
        if self.filtered_code_list is None:
            stock_dict = StockAcquisitionFacade(update=update, process=process, filter=self.universe_filter).get_stock_data_dict()
        else:
            print('filtered_code_listが設定されているため、universe_filterの設定を無視します。')
            stock_dict = StockAcquisitionFacade(update=update, process=process, filtered_code_list=self.filtered_code_list).get_stock_data_dict()
        if update:
            fu = FeaturesUpdateFacade()
            await fu.update_all()
        return stock_dict


if __name__ == '__main__':
    import asyncio
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    
    async def main():
        daf = DataUpdateFacade('update_and_load', universe_filter=universe_filter)
        stock_dict = await daf.execute()
        print(stock_dict)
    
    asyncio.get_event_loop().run_until_complete(main())