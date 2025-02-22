from trading.sbi.operations.order_manager import NewOrderManager
from trading.sbi_trading_logic import StockSelector, NewOrderMaker

from trading.sbi import LoginHandler, TradePossibilityManager, OrderManager
from models import MLDataset
from utils.paths import Paths

async def main():
    dataset_folder_path = f'{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250125'
    dataset = MLDataset(dataset_folder_path)
    lh = LoginHandler()
    tpm = TradePossibilityManager(lh)
    sector_dif_csv = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv'
    stock_selector = StockSelector(dataset.stock_selection_materials.order_price_df,
                                dataset.stock_selection_materials.pred_result_df,
                                tpm,
                                None,
                                sector_dif_csv
                                )
    long_df, short_df, pred_sector = await stock_selector.select(margin_power=60000000)
    om = NewOrderManager(lh)
    nom = NewOrderMaker(long_df, short_df, om)
    failed_orders = await nom.run_new_orders()


import asyncio

asyncio.get_event_loop().run_until_complete(main())