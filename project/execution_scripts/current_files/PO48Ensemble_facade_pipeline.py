from __future__ import annotations

import os
from datetime import datetime
import asyncio

from utils.notifier import SlackNotifier
from utils.paths import Paths
from facades import (
    DataUpdateFacade,
    MachineLearningFacade,
    OrderExecutionFacade,
    TradeDataFacade,
)
from trading import TradingFacade


async def main() -> None:
    slack = SlackNotifier(program_name=os.path.basename(__file__))
    slack.start(message='プログラムを開始します。', should_send_program_name=True)

    # パラメータ設定
    sector_redef_csv = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv"
    sector_index_parquet = f"{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet"
    dataset_path1 = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250607"
    dataset_path2 = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LightGBMlearned_in_250607"
    ensembled_dataset_path = f"{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250607"
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    trading_sector_num = 3
    candidate_sector_num = 5
    top_slope = 1
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2024, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31)

    try:
        # 1. データ更新
        data_facade = DataUpdateFacade('update_and_load', universe_filter)
        stock_dict = await data_facade.execute()

        # 2. 機械学習
        ml_facade = MachineLearningFacade(
            'train_and_predict',
            dataset_path1,
            dataset_path2,
            ensembled_dataset_path,
            sector_redef_csv,
            sector_index_parquet,
            train_start_day,
            train_end_day,
            test_start_day,
            test_end_day,
        )
        ml_dataset = await ml_facade.execute(stock_dict)

        # 3. 発注
        trading_facade = TradingFacade()
        order_facade = OrderExecutionFacade(
            'new',
            trading_facade,
            sector_redef_csv,
            trading_sector_num,
            candidate_sector_num,
            top_slope,
        )
        await order_facade.execute(ml_dataset)

        # 4. 取引データの取得
        trade_data_facade = TradeDataFacade('fetch', trading_facade, sector_redef_csv)
        await trade_data_facade.execute()

        slack.finish(message='すべての処理が完了しました。')
    except Exception:
        from utils.error_handler import error_handler
        error_handler.handle_exception(Paths.ERROR_LOG_CSV)
        error_handler.handle_exception(Paths.ERROR_LOG_BACKUP)
        slack.send_error_log(f"エラーが発生しました。\n詳細は{Paths.ERROR_LOG_CSV}を確認してください。")


if __name__ == '__main__':
    asyncio.get_event_loop().run_until_complete(main())

