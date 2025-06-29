from __future__ import annotations

import os
import asyncio
from datetime import datetime

from utils.notifier import SlackNotifier
from utils.paths import Paths
from facades import (
    DataUpdateFacade,
    MachineLearningFacade,
    OrderExecutionFacade,
    TradeDataFacade,
    LassoLearningFacade,
    ModeForStrategy,
    ModelOrderConfig,
)
from trading import TradingFacade

async def main() -> None:
    slack = SlackNotifier(program_name=os.path.basename(__file__))
    slack.start(message="プログラムを開始します。", should_send_program_name=True)

    # パラメータ設定
    # 54業種LASSO用設定
    datasets_54_lasso = f"{Paths.ML_DATASETS_FOLDER}/54sectors_LASSO_learned_in_250623"
    sector_csv_54_lasso = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/54sectors_2024-2025.csv"
    trading_sector_num_54_lasso = 2
    candidate_sector_num_54_lasso = 4
    top_slope_54_lasso = 1

    # 48業種LASSO+LightGBMアンサンブル用設定
    sector_csv_48_ensemble = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv"
    sector_index_parquet = f"{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet"
    datasets_path1 = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250615"
    datasets_path2 = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LightGBMlearned_in_250615"
    ensembled_datasets_path = f"{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250615"
    model1_weight = 6.7
    model2_weight = 1.3
    trading_sector_num_48_ensemble = 2
    candidate_sector_num_48_ensemble = 4
    top_slope_48_ensemble = 1

    # 各モデルへの資金配分
    fund_allocation_54_lasso = 0.67
    fund_allocation_48_ensemble = 0.33

    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))"
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2021, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31)

    try:
        modes = ModeForStrategy.generate_mode()

        # 1. データ更新
        data_facade = DataUpdateFacade(
            mode=modes.data_update_mode,
            universe_filter=universe_filter,
        )
        stock_dict = await data_facade.execute()

        # 2. 機械学習
        ml_54_lasso = LassoLearningFacade(
            mode=modes.machine_learning_mode,
            dataset_path=datasets_54_lasso,
        ).execute()

        ml_48_ensemble = MachineLearningFacade(
            mode=modes.machine_learning_mode,
            stock_dfs_dict=stock_dict,
            sector_redef_csv_path=sector_csv_48_ensemble,
            sector_index_parquet_path=sector_index_parquet,
            datasets1_path=datasets_path1,
            datasets2_path=datasets_path2,
            ensembled_datasets_path=ensembled_datasets_path,
            model1_weight=model1_weight,
            model2_weight=model2_weight,
            train_start_day=train_start_day,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
        ).execute()

        # 3. 発注
        trade_facade = TradingFacade()
        configs = [
            ModelOrderConfig(
                ml_datasets=ml_54_lasso,
                sector_csv=sector_csv_54_lasso,
                trading_sector_num=trading_sector_num_54_lasso,
                candidate_sector_num=candidate_sector_num_54_lasso,
                top_slope=top_slope_54_lasso,
                margin_weight=fund_allocation_54_lasso,
            ),
            ModelOrderConfig(
                ml_datasets=ml_48_ensemble,
                sector_csv=sector_csv_48_ensemble,
                trading_sector_num=trading_sector_num_48_ensemble,
                candidate_sector_num=candidate_sector_num_48_ensemble,
                top_slope=top_slope_48_ensemble,
                margin_weight=fund_allocation_48_ensemble,
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
