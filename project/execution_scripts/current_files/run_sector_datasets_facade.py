# -*- coding: utf-8 -*-
"""SectorMLDatasetsFacade を実行して学習用データセットを作成するスクリプト."""

from datetime import datetime

from facades import SectorMLDatasetsFacade
from utils.paths import Paths
from acquisition.jquants_api_operations.facades import StockAcquisitionFacade


def main() -> None:
    """SectorMLDatasetsFacade の実行."""
    # PO48Ensemble_fixed_241014.py を参考にした各種パラメータ
    sector_redefinitions_csv = (
        f"{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv"
    )
    sector_index_parquet = (
        f"{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet"
    )
    dataset_root = f"{Paths.ML_DATASETS_FOLDER}/48sectors_learned_in_250603"

    universe_filter = (
        "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))"
    )

    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2024, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31)

    # 株価データの取得
    stock_dfs_dict = StockAcquisitionFacade(filter=universe_filter).get_stock_data_dict()

    facade = SectorMLDatasetsFacade()
    facade.create_datasets(
        stock_dfs_dict=stock_dfs_dict,
        sector_redefinitions_csv=sector_redefinitions_csv,
        sector_index_parquet=sector_index_parquet,
        dataset_root=dataset_root,
        train_start_day=train_start_day,
        train_end_day=train_end_day,
        test_start_day=test_start_day,
        test_end_day=test_end_day,
        outlier_threshold=3,
    )


if __name__ == "__main__":
    main()
