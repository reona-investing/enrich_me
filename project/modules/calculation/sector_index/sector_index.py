#%% モジュールのインポート
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
import pandas as pd
from typing import Tuple

from .calculators.marketcap_calculator import MarketCapCalculator
from .preparers.sector_data_preparer import SectorDataPreparer
from .calculators.index_calculator import SectorIndexCalculator


class SectorIndex:
    """Stock informationからセクターインデックスを計算するメインクラス。"""

    def __init__(self, stock_dfs_dict: dict, sector_redefinitions_csv: str, sector_index_parquet: str):
        """各種データのパスとデータフレームを受け取り初期化を行う。

        Args:
            stock_dfs_dict (dict): ``'price'`` および ``'fin'`` をキーにもつデータフレーム辞書。
            sector_redefinitions_csv (str): セクター再定義 CSV へのパス。
            sector_index_parquet (str): 計算結果を保存する parquet ファイルのパス。
        """

        self.stock_dfs_dict = stock_dfs_dict
        self.sector_redefinitions_csv = sector_redefinitions_csv
        self.sector_index_parquet = sector_index_parquet
        self.sector_index_df: pd.DataFrame | None = None
        self.order_price_df: pd.DataFrame | None = None
        self.marketcap_df: pd.DataFrame | None = None
        self.sector_index_dict: dict[str, pd.DataFrame] | None = None
        self.order_price_dict: dict[str, pd.DataFrame] | None = None
        self._marketcap_calc = MarketCapCalculator()
        self._data_preparer = SectorDataPreparer()
        self._index_calc = SectorIndexCalculator()

    _col_names = None
    
    @staticmethod
    def _get_column_names() -> Tuple[dict, dict, dict]:
        if SectorIndex._col_names is None:
            fin_col_configs = ColumnConfigsGetter(Paths.STOCK_FIN_COLUMNS_YAML)
            fin_cols = fin_col_configs.get_all_columns_name_asdict()
            price_col_configs = ColumnConfigsGetter(Paths.STOCK_PRICE_COLUMNS_YAML)
            price_cols = price_col_configs.get_all_columns_name_asdict()
            sector_col_configs = ColumnConfigsGetter(Paths.SECTOR_INDEX_COLUMNS_YAML)
            sector_cols = sector_col_configs.get_all_columns_name_asdict()
            
            SectorIndex._col_names = (fin_cols, price_cols, sector_cols)
        return SectorIndex._col_names

    # ========================================
    # パブリックメソッド（外部から呼び出し可能）
    # ========================================

    def calc_sector_index(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """セクターインデックスを算出して ``order_price_df`` も返す。

        ``stock_dfs_dict`` で渡された株価データと財務データを用いて
        時価総額を計算し、CSV で指定されたセクター定義を結合して
        セクターごとの指数を計算する。結果は ``sector_index_parquet``
        に保存される。

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]:
                計算したセクターインデックスと発注処理用株価データ。
        """

        if self.sector_index_df is not None and self.order_price_df is not None:
            return self.sector_index_df, self.order_price_df

        if (
            self.stock_dfs_dict is None
            or self.sector_redefinitions_csv is None
            or self.sector_index_parquet is None
        ):
            raise ValueError(
                "stock_dfs_dict, sector_redefinitions_csv and sector_index_parquet must be set"
            )

        # セクターインデックスを算出
        _, price_col, sector_col = SectorIndex._get_column_names()
        stock_price = self.stock_dfs_dict['price']
        stock_fin = self.stock_dfs_dict['fin']

        # 価格情報に発行済み株式数の情報を結合
        order_price_df = self.calc_marketcap(stock_price, stock_fin)

        # セクター定義を読み込み、株価データと結合
        sector_price_data = self._data_preparer.from_csv(
            order_price_df,
            self.sector_redefinitions_csv,
            self._get_column_names,
        )

        # 共通のインデックス計算処理
        new_sector_price = self._index_calc.aggregate(sector_price_data, self._get_column_names)
        
        # データフレームを保存して、インデックスを設定
        new_sector_price = new_sector_price.reset_index()
        new_sector_price.to_parquet(self.sector_index_parquet)
        new_sector_price = new_sector_price.set_index([sector_col['日付'], sector_col['セクター']])

        order_price_df = order_price_df[[
            sector_col['日付'], sector_col['銘柄コード'], sector_col['終値時価総額'], 
            sector_col['終値'], price_col['取引高']
        ]]
        self.sector_index_df = new_sector_price
        self.order_price_df = order_price_df
        print('セクターのインデックス値の算出が完了しました。')
        
        return new_sector_price, order_price_df

    def calc_sector_index_by_dict(self, sector_stock_dict: dict, stock_price_data: pd.DataFrame) -> pd.DataFrame:
        """辞書形式のセクター定義からインデックスを計算する。

        Args:
            sector_stock_dict (dict):
                セクター名をキー、銘柄コードのリストを値とする辞書。
                例: ``{"JPY感応株": ["6758", "6501"], "JPbond感応株": ["6751", ...]}``
            stock_price_data (pd.DataFrame):
                ``calc_marketcap`` の結果と同じ形式の株価データ。

        Returns:
            pd.DataFrame: セクターインデックスのデータフレーム。
        """
        # セクター定義を辞書から作成し、株価データと結合
        sector_price_data = self._data_preparer.from_dict(stock_price_data, sector_stock_dict, self._get_column_names)

        # 共通のインデックス計算処理
        sector_index = self._index_calc.aggregate(sector_price_data, self._get_column_names)
        self.sector_index_df = sector_index

        return sector_index

    def get_sector_index_dict(self) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
        """セクター別のデータフレーム辞書を取得する。

        :meth:`calc_sector_index` が未実行の場合は内部で呼び出して
        計算結果をキャッシュする。

        Returns
        -------
        tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]
            セクターインデックスと発注処理用の価格データを、それぞれセクター別に分割した辞書を返す。
        """

        if self.sector_index_dict is not None and self.order_price_dict is not None:
            return self.sector_index_dict, self.order_price_dict

        if self.sector_index_df is None or self.order_price_df is None:
            self.calc_sector_index()

        sector_index_df = self.sector_index_df
        order_price_df = self.order_price_df

        if sector_index_df is None or not isinstance(sector_index_df.index, pd.MultiIndex):
            raise ValueError("sector_index_df must be MultiIndex")

        _, _, sector_col = SectorIndex._get_column_names()
        sector_name = sector_col['セクター']
        date_name = sector_col['日付']

        sector_dict: dict[str, pd.DataFrame] = {}
        for sec in sector_index_df.index.get_level_values(sector_name).unique():
            df = sector_index_df.xs(sec, level=sector_name)
            df.index.name = date_name
            sector_dict[sec] = df

        # 発注用株価データをセクター別に分割
        order_price_data = self._data_preparer.from_csv(
            order_price_df,
            self.sector_redefinitions_csv,
            self._get_column_names,
        )
        order_dict: dict[str, pd.DataFrame] = {}
        for sec in order_price_data[sector_name].unique():
            df = order_price_data[order_price_data[sector_name] == sec].copy()
            df = df.set_index([date_name, sector_col['銘柄コード']])
            order_dict[sec] = df

        self.sector_index_dict = sector_dict
        self.order_price_dict = order_dict

        return sector_dict, order_dict

    def calc_marketcap(self, stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """各銘柄の日次時価総額を計算する。

        Args:
            stock_price (pd.DataFrame): 株価データ。
            stock_fin (pd.DataFrame): 財務データ。

        Returns:
            pd.DataFrame: 時価総額および補正値を付加した株価データ。
        """
        # 価格情報に発行済み株式数の情報を照合
        stock_price_with_shares = self._marketcap_calc.merge_stock_price_and_shares(stock_price, stock_fin, self._get_column_names)
        # 発行済み株式数の補正係数を算出
        stock_price_cap = self._marketcap_calc.calc_adjustment_factor(stock_price_with_shares, stock_price, self._get_column_names)
        stock_price_cap = self._marketcap_calc.adjust_shares(stock_price_cap, self._get_column_names)
        # 時価総額と指数計算用の補正値を算出
        stock_price_cap = self._marketcap_calc.calc_marketcap(stock_price_cap, self._get_column_names)
        stock_price_cap = self._marketcap_calc.calc_correction_value(stock_price_cap, self._get_column_names)
        self.marketcap_df = stock_price_cap
        
        return stock_price_cap
    
if __name__ == '__main__':
    from acquisition.jquants_api_operations.facades import StockAcquisitionFacade
    acq = StockAcquisitionFacade(filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))")
    stock_dfs = acq.get_stock_data_dict()

    sic = SectorIndex(
        stock_dfs_dict=stock_dfs,
        sector_redefinitions_csv=f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/topix1000.csv',
        sector_index_parquet=f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/TOPIX1000_price.parquet',
    )
    # セクターインデックスを計算し、セクター別の辞書を取得
    sector_price_df, order_price_df = sic.calc_sector_index()
    index_dict, order_dict = sic.get_sector_index_dict()
    #print(sector_price_df.index.get_level_values('Sector').unique())
    print(sector_price_df)

    marketcap_df = sic.calc_marketcap(stock_dfs['price'], stock_dfs['fin'])
    sector_index = sic.calc_sector_index_by_dict(sector_stock_dict={'JPY+': ['2413', '3141', '4587', '1835', '4684'],
                                                                 'JPY-': ['7283', '7296', '5988', '8015', '7278']},
                                                                 stock_price_data=marketcap_df)
    
    print(sector_index)