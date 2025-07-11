from __future__ import annotations

import os
import asyncio
from datetime import datetime

from utils.notifier import SlackNotifier
from utils.paths import Paths
from facades import (
    DataUpdateFacade,
    OrderExecutionFacade,
    TradeDataFacade,
    LassoLearningFacade,
    SubseqLgbmLearningFacade,
    RankEnsembleFacade,
    ModeForStrategy,
    ModelOrderConfig,
)
from trading import TradingFacade

async def main() -> None:
    slack = SlackNotifier(program_name=os.path.basename(__file__))
    slack.start(message="プログラムを開始します。", should_send_program_name=True)

    # パラメータ設定
    # 56業種LASSO用設定
    dataset_56_lasso = f"{Paths.ML_DATASETS_FOLDER}/56sectors_LASSO_learned_in_250702"
    sector_csv_56_lasso = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/56sectors_2024-2025.csv"
    sector_index_56_parquet = f"{Paths.SECTOR_PRICE_FOLDER}/56sectors_price.parquet"
    trading_sector_num_56_lasso = 2
    candidate_sector_num_56_lasso = 4
    top_slope_56_lasso = 1

    # 48業種LASSO+LightGBMアンサンブル用設定
    dataset_48_lasso = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LASSO_learned_in_250702"
    dataset_48_lgbm = f"{Paths.ML_DATASETS_FOLDER}/48sectors_LightGBMlearned_in_250702"
    sector_csv_48 = f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv"
    sector_index_48_parquet = f"{Paths.SECTOR_PRICE_FOLDER}/48sectors_price.parquet"
    trading_sector_num_48 = 2
    candidate_sector_num_48 = 4
    top_slope_48_lgbm = 1

    # 各モデルへの資金配分
    fund_allocation_56_lasso = 0.67
    fund_allocation_48_lgbm = 0.33

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
        ml_56_lasso = LassoLearningFacade(
            mode=modes.machine_learning_mode,
            stock_dfs_dict=stock_dict,
            dataset_path=dataset_56_lasso,
            sector_redef_csv_path=sector_csv_56_lasso,
            sector_index_parquet_path=sector_index_56_parquet,
            train_start_day=train_start_day,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
        ).execute()

        ml_48_lasso = LassoLearningFacade(
            mode=modes.machine_learning_mode,
            stock_dfs_dict=stock_dict,
            dataset_path=dataset_48_lasso,
            sector_redef_csv_path=sector_csv_48,
            sector_index_parquet_path=sector_index_48_parquet,
            train_start_day=train_start_day,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
        ).execute()

        ml_48_lgbm = SubseqLgbmLearningFacade(
            preliminary_model=ml_48_lasso,
            mode=modes.machine_learning_mode,
            stock_dfs_dict=stock_dict,
            dataset_path=dataset_48_lgbm,
            sector_redef_csv_path=sector_csv_48,
            sector_index_parquet_path=sector_index_48_parquet,
            train_start_day=train_start_day,
            train_end_day=train_end_day,
            test_start_day=test_start_day,
            test_end_day=test_end_day,
        ).execute()

        # 3. 発注
        configs = [
            ModelOrderConfig(
                ml_dataset=ml_56_lasso,
                sector_csv=sector_csv_56_lasso,
                trading_sector_num=trading_sector_num_56_lasso,
                candidate_sector_num=candidate_sector_num_56_lasso,
                top_slope=top_slope_56_lasso,
                margin_weight=fund_allocation_56_lasso,
            ),
            ModelOrderConfig(
                ml_dataset=ml_48_lgbm,
                sector_csv=sector_csv_48,
                trading_sector_num=trading_sector_num_48,
                candidate_sector_num=candidate_sector_num_48,
                top_slope=top_slope_48_lgbm,
                margin_weight=fund_allocation_48_lgbm,
            ),
        ]

        trade_facade = TradingFacade()
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