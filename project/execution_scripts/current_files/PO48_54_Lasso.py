from __future__ import annotations

import os
import asyncio

from utils.notifier import SlackNotifier
from utils.paths import Paths
from facades import (
    DataUpdateFacade,
    ModeForStrategy,
    TradeDataFacade,
    LassoLearningFacade,
    OrderExecutionFacade,
    ModelOrderConfig,
)
from trading import TradingFacade

async def main() -> None:
    slack = SlackNotifier(program_name=os.path.basename(__file__))
    slack.start(message="プログラムを開始します。", should_send_program_name=True)

    # パラメータ設定
    datasets_48 = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250615"
    datasets_54 = f"{Paths.ML_DATASETS_FOLDER}/54sectors_LASSO_learned_in_250623"
    sector_csv_48 = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv"
    sector_csv_54 = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/54sectors_2024-2025.csv"
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))"
    trading_sector_num = 2
    candidate_sector_num = 4
    top_slope = 1

    try:
        modes = ModeForStrategy.generate_mode()

        # 1. データ更新
        data_facade = DataUpdateFacade(
            mode=modes.data_update_mode, universe_filter=universe_filter
        )
        await data_facade.execute()

        # 2. 機械学習
        ml_48 = LassoLearningFacade(
            mode=modes.machine_learning_mode,
            dataset_path=datasets_48,
        ).execute()
        ml_54 = LassoLearningFacade(
            mode=modes.machine_learning_mode,
            dataset_path=datasets_54,
        ).execute()

        # 3. 発注
        trade_facade = TradingFacade()
        configs = [
            ModelOrderConfig(
                ml_datasets=ml_48,
                sector_csv=sector_csv_48,
                trading_sector_num=trading_sector_num,
                candidate_sector_num=candidate_sector_num,
                top_slope=top_slope,
                margin_weight=0.5,
            ),
            ModelOrderConfig(
                ml_datasets=ml_54,
                sector_csv=sector_csv_54,
                trading_sector_num=trading_sector_num,
                candidate_sector_num=candidate_sector_num,
                top_slope=top_slope,
                margin_weight=0.5,
            ),
        ]

        order_facade = OrderExecutionFacade(
            mode=modes.order_execution_mode,
            trade_facade=trade_facade,
        )
        orders_df = await order_facade.execute(configs)
        if orders_df is not None:
            orders_df.to_csv(Paths.ORDERS_CSV, index=False)

        # 4. 取引データの取得
        trade_data_facade = TradeDataFacade(
            mode=modes.trade_data_fetch_mode,
            trade_facade=trade_facade,
            sector_csv=sector_csv_48,
        )
        await trade_data_facade.execute()
        
        slack.finish(message="すべての処理が完了しました。")
    except Exception:
        from utils.error_handler import error_handler

        error_handler.handle_exception(Paths.ERROR_LOG_CSV)
        slack.send_error_log(
            f"エラーが発生しました。\n詳細は{Paths.ERROR_LOG_CSV}を確認してください。"
        )


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())