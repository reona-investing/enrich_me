#%% モジュールのインポート
from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Tuple, Optional, Dict


class SectorIndex:
    """
    セクターインデックス計算クラス
    
    入力データと計算結果をプロパティとして保持し、
    オブジェクト指向の原則に従った設計
    """
    
    _col_names = None
    
    def __init__(self):
        """初期化"""
        # 入力データのプロパティ
        self._stock_dfs_dict: Optional[Dict[str, pd.DataFrame]] = None
        self._sector_definitions: Optional[pd.DataFrame] = None
        self._sector_stock_dict: Optional[Dict[str, list]] = None
        
        # 中間計算結果のプロパティ
        self._stock_price_with_marketcap: Optional[pd.DataFrame] = None
        self._sector_price_data: Optional[pd.DataFrame] = None
        
        # 最終結果のプロパティ
        self._sector_index: Optional[pd.DataFrame] = None
        self._stock_price_for_order: Optional[pd.DataFrame] = None
        
        # カラム名の取得
        self._fin_cols, self._price_cols, self._sector_cols = self._get_column_names()
    
    # ========================================
    # プロパティ（入力データ）
    # ========================================
    
    @property
    def stock_dfs_dict(self) -> Optional[Dict[str, pd.DataFrame]]:
        """株価・財務データ辞書"""
        return self._stock_dfs_dict
    
    @stock_dfs_dict.setter
    def stock_dfs_dict(self, value: Dict[str, pd.DataFrame]):
        """株価・財務データ辞書の設定"""
        required_keys = {'list', 'fin', 'price'}
        if not all(key in value for key in required_keys):
            raise ValueError(f"辞書には {required_keys} のキーが必要です")
        self._stock_dfs_dict = value
        # 入力データ変更時は計算結果をクリア
        self._clear_calculation_results()
    
    @property
    def sector_definitions(self) -> Optional[pd.DataFrame]:
        """セクター定義データフレーム"""
        return self._sector_definitions
    
    @sector_definitions.setter
    def sector_definitions(self, value: pd.DataFrame):
        """セクター定義データフレームの設定"""
        self._sector_definitions = value
        self._clear_calculation_results()
    
    @property
    def sector_stock_dict(self) -> Optional[Dict[str, list]]:
        """セクター銘柄辞書"""
        return self._sector_stock_dict
    
    @sector_stock_dict.setter
    def sector_stock_dict(self, value: Dict[str, list]):
        """セクター銘柄辞書の設定"""
        self._sector_stock_dict = value
        self._clear_calculation_results()
    
    # ========================================
    # プロパティ（計算結果）
    # ========================================
    
    @property
    def stock_price_with_marketcap(self) -> Optional[pd.DataFrame]:
        """時価総額付き株価データ"""
        return self._stock_price_with_marketcap
    
    @property
    def sector_price_data(self) -> Optional[pd.DataFrame]:
        """セクター情報付き株価データ"""
        return self._sector_price_data
    
    @property
    def sector_index(self) -> Optional[pd.DataFrame]:
        """セクターインデックス"""
        return self._sector_index
    
    @property
    def stock_price_for_order(self) -> Optional[pd.DataFrame]:
        """発注用株価データ"""
        return self._stock_price_for_order
    
    # ========================================
    # パブリックメソッド
    # ========================================
    
    def load_from_csv(self, sector_redefinitions_csv: str) -> 'SectorIndex':
        """
        CSVファイルからセクター定義を読み込み
        
        Args:
            sector_redefinitions_csv: セクター定義CSVファイルパス
            
        Returns:
            self: メソッドチェーン用
        """
        sector_def = pd.read_csv(sector_redefinitions_csv).dropna(how='any', axis=1)
        sector_def[self._sector_cols['銘柄コード']] = sector_def[self._sector_cols['銘柄コード']].astype(str)
        self.sector_definitions = sector_def
        return self
    
    def load_from_dict(self, sector_stock_dict: Dict[str, list]) -> 'SectorIndex':
        """
        辞書からセクター定義を読み込み
        
        Args:
            sector_stock_dict: セクター銘柄辞書
            
        Returns:
            self: メソッドチェーン用
        """
        self.sector_stock_dict = sector_stock_dict
        return self
    
    def calculate_marketcap(self) -> 'SectorIndex':
        """
        時価総額を計算
        
        Returns:
            self: メソッドチェーン用
        """
        if self._stock_dfs_dict is None:
            raise ValueError("stock_dfs_dict が設定されていません")
        
        stock_price = self._stock_dfs_dict['price']
        stock_fin = self._stock_dfs_dict['fin']
        
        # 価格情報に発行済み株式数の情報を結合
        stock_price_with_shares = self._merge_stock_price_and_shares(stock_price, stock_fin)
        
        # 発行済み株式数の補正係数を算出
        stock_price_cap = self._calc_adjustment_factor(stock_price_with_shares, stock_price)
        stock_price_cap = self._adjust_shares(stock_price_cap)
        
        # 時価総額と指数計算用の補正値を算出
        stock_price_cap = self._calc_marketcap(stock_price_cap)
        stock_price_cap = self._calc_correction_value(stock_price_cap)
        
        self._stock_price_with_marketcap = stock_price_cap
        return self
    
    def prepare_sector_data(self) -> 'SectorIndex':
        """
        セクター情報付き株価データを準備
        
        Returns:
            self: メソッドチェーン用
        """
        if self._stock_price_with_marketcap is None:
            raise ValueError("時価総額計算が完了していません。calculate_marketcap()を先に実行してください")
        
        if self._sector_definitions is not None:
            # CSV定義を使用
            self._sector_price_data = pd.merge(
                self._sector_definitions, 
                self._stock_price_with_marketcap, 
                how='right', 
                on=self._sector_cols['銘柄コード']
            )
        elif self._sector_stock_dict is not None:
            # 辞書定義を使用
            sector_definitions = []
            for sector_name, stock_codes in self._sector_stock_dict.items():
                for code in stock_codes:
                    sector_definitions.append({
                        self._sector_cols['銘柄コード']: str(code),
                        self._sector_cols['セクター']: sector_name
                    })
            
            sector_df = pd.DataFrame(sector_definitions)
            self._sector_price_data = pd.merge(
                sector_df, 
                self._stock_price_with_marketcap, 
                how='inner', 
                on=self._sector_cols['銘柄コード']
            )
        else:
            raise ValueError("セクター定義が設定されていません")
        
        return self
    
    def calculate_sector_index(self) -> 'SectorIndex':
        """
        セクターインデックスを計算
        
        Returns:
            self: メソッドチェーン用
        """
        if self._sector_price_data is None:
            raise ValueError("セクターデータの準備が完了していません。prepare_sector_data()を先に実行してください")
        
        # セクターごとに集計
        columns_to_sum = [
            self._sector_cols['始値時価総額'], self._sector_cols['終値時価総額'], 
            self._sector_cols['高値時価総額'], self._sector_cols['安値時価総額'], 
            self._sector_cols['発行済み株式数'], self._sector_cols['指数算出用の補正値']
        ]
        
        sector_index = self._sector_price_data.groupby(
            [self._sector_cols['日付'], self._sector_cols['セクター']]
        )[columns_to_sum].sum()
        
        # 1日リターンの計算
        sector_index[self._sector_cols['1日リターン']] = sector_index[self._sector_cols['終値時価総額']] / (
            sector_index.groupby(self._sector_cols['セクター'])[self._sector_cols['終値時価総額']].shift(1) + 
            sector_index[self._sector_cols['指数算出用の補正値']]
        ) - 1

        sector_index[self._sector_cols['終値前日比']] = 1 + sector_index[self._sector_cols['1日リターン']]
        
        # 終値の計算（累積積）
        sector_index[self._sector_cols['終値']] = sector_index.groupby(
            self._sector_cols['セクター']
        )[self._sector_cols['終値前日比']].cumprod()
        
        # OHLC計算
        sector_index = self._calculate_ohlc(sector_index)
        
        self._sector_index = sector_index
        return self
    
    def prepare_order_data(self) -> 'SectorIndex':
        """
        発注用データを準備
        
        Returns:
            self: メソッドチェーン用
        """
        if self._stock_price_with_marketcap is None:
            raise ValueError("時価総額計算が完了していません")
        
        self._stock_price_for_order = self._stock_price_with_marketcap[[
            self._sector_cols['日付'], self._sector_cols['銘柄コード'], 
            self._sector_cols['終値時価総額'], self._sector_cols['終値'], 
            self._price_cols['取引高']
        ]]
        return self
    
    def save_sector_index(self, output_path: str) -> 'SectorIndex':
        """
        セクターインデックスを保存
        
        Args:
            output_path: 出力ファイルパス
            
        Returns:
            self: メソッドチェーン用
        """
        if self._sector_index is None:
            raise ValueError("セクターインデックスが計算されていません")
        
        # データフレームを保存して、インデックスを設定
        sector_index_reset = self._sector_index.reset_index()
        sector_index_reset.to_parquet(output_path)
        self._sector_index = sector_index_reset.set_index(
            [self._sector_cols['日付'], self._sector_cols['セクター']]
        )
        return self
    
    def execute_full_calculation_from_csv(
        self, 
        stock_dfs_dict: Dict[str, pd.DataFrame], 
        sector_redefinitions_csv: str, 
        output_path: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        CSVからの完全な計算実行
        
        Args:
            stock_dfs_dict: 株価・財務データ辞書
            sector_redefinitions_csv: セクター定義CSVファイルパス
            output_path: 出力ファイルパス
            
        Returns:
            セクターインデックス, 発注用データ
        """
        self.stock_dfs_dict = stock_dfs_dict
        
        (self.load_from_csv(sector_redefinitions_csv)
         .calculate_marketcap()
         .prepare_sector_data()
         .calculate_sector_index()
         .prepare_order_data()
         .save_sector_index(output_path))
        
        print('セクターのインデックス値の算出が完了しました。')
        return self._sector_index, self._stock_price_for_order
    
    def execute_calculation_from_dict(
        self, 
        sector_stock_dict: Dict[str, list], 
        stock_price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        辞書からの計算実行
        
        Args:
            sector_stock_dict: セクター銘柄辞書
            stock_price_data: 時価総額計算済み株価データ
            
        Returns:
            セクターインデックス
        """
        self._stock_price_with_marketcap = stock_price_data
        
        (self.load_from_dict(sector_stock_dict)
         .prepare_sector_data()
         .calculate_sector_index())
        
        return self._sector_index
    
    # ========================================
    # プライベートメソッド
    # ========================================
    
    def _clear_calculation_results(self):
        """計算結果をクリア"""
        self._stock_price_with_marketcap = None
        self._sector_price_data = None
        self._sector_index = None
        self._stock_price_for_order = None
    
    @staticmethod
    def _get_column_names() -> Tuple[dict, dict, dict]:
        """カラム名設定を取得"""
        if SectorIndex._col_names is None:
            fin_col_configs = ColumnConfigsGetter(Paths.STOCK_FIN_COLUMNS_YAML)
            fin_cols = fin_col_configs.get_all_columns_name_asdict()
            price_col_configs = ColumnConfigsGetter(Paths.STOCK_PRICE_COLUMNS_YAML)
            price_cols = price_col_configs.get_all_columns_name_asdict()
            sector_col_configs = ColumnConfigsGetter(Paths.SECTOR_INDEX_COLUMNS_YAML)
            sector_cols = sector_col_configs.get_all_columns_name_asdict()
            
            SectorIndex._col_names = (fin_cols, price_cols, sector_cols)
        return SectorIndex._col_names
    
    def _calculate_ohlc(self, sector_index: pd.DataFrame) -> pd.DataFrame:
        """始値、高値、安値の計算"""
        sector_index[self._sector_cols['始値']] = (
            sector_index[self._sector_cols['終値']] * 
            sector_index[self._sector_cols['始値時価総額']] / 
            sector_index[self._sector_cols['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        sector_index[self._sector_cols['高値']] = (
            sector_index[self._sector_cols['終値']] * 
            sector_index[self._sector_cols['高値時価総額']] / 
            sector_index[self._sector_cols['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        sector_index[self._sector_cols['安値']] = (
            sector_index[self._sector_cols['終値']] * 
            sector_index[self._sector_cols['安値時価総額']] / 
            sector_index[self._sector_cols['終値時価総額']]
        ).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        
        return sector_index
    
    def _merge_stock_price_and_shares(self, stock_price: pd.DataFrame, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """期末日以降最初の営業日時点での発行済株式数を結合"""
        business_days = stock_price[self._price_cols['日付']].unique()
        shares_df = self._calc_shares_at_end_period(stock_fin)
        shares_df = self._append_next_period_start_date(shares_df, business_days)
        merged_df = self._merge_with_stock_price(stock_price, shares_df)
        return merged_df
    
    def _calc_shares_at_end_period(self, stock_fin: pd.DataFrame) -> pd.DataFrame:
        """期末日時点での発行済株式数を計算"""
        shares_df = stock_fin[[
            self._fin_cols['銘柄コード'], self._fin_cols['日付'], 
            self._fin_cols['発行済み株式数'], self._fin_cols['当会計期間終了日']
        ]].copy()
        shares_df = shares_df.sort_values(self._fin_cols['日付']).drop(self._fin_cols['日付'], axis=1)
        shares_df = shares_df.drop_duplicates(
            subset=[self._fin_cols['当会計期間終了日'], self._fin_cols['銘柄コード']], 
            keep='last'
        )
        shares_df['NextPeriodStartDate'] = pd.to_datetime(shares_df[self._fin_cols['当会計期間終了日']]) + timedelta(days=1)
        shares_df['isSettlementDay'] = True
        return shares_df
    
    def _append_next_period_start_date(self, shares_df: pd.DataFrame, business_days: np.ndarray) -> pd.DataFrame:
        """次期開始日を営業日ベースで計算"""
        shares_df['NextPeriodStartDate'] = shares_df['NextPeriodStartDate'].apply(
            self._find_next_business_day, business_days=business_days
        )
        return shares_df
    
    @staticmethod
    def _find_next_business_day(date: pd.Timestamp, business_days: np.ndarray) -> pd.Timestamp | Any:
        """任意の日付から翌営業日を探す"""
        if pd.isna(date):
            return date
        while date not in business_days:
            date += np.timedelta64(1, 'D')
        return date
    
    def _merge_with_stock_price(self, stock_price: pd.DataFrame, shares_df: pd.DataFrame) -> pd.DataFrame:
        """価格データに発行済株式数情報を結合"""
        stock_price = stock_price.rename(columns={
            self._price_cols['銘柄コード']: self._sector_cols['銘柄コード'], 
            self._price_cols['日付']: self._sector_cols['日付']
        })
        shares_df = shares_df.rename(columns={
            self._fin_cols['銘柄コード']: self._sector_cols['銘柄コード'], 
            'NextPeriodStartDate': self._sector_cols['日付']
        })

        merged_df = pd.merge(
            stock_price, 
            shares_df[[
                self._sector_cols['日付'], self._sector_cols['銘柄コード'], 
                self._fin_cols['発行済み株式数'], 'isSettlementDay'
            ]],
            on=[self._sector_cols['日付'], self._sector_cols['銘柄コード']],
            how='left'
        )
        merged_df = merged_df.rename(columns={
            self._fin_cols['発行済み株式数']: self._sector_cols['発行済み株式数'],
            self._price_cols['始値']: self._sector_cols['始値'],
            self._price_cols['終値']: self._sector_cols['終値'],
            self._price_cols['高値']: self._sector_cols['高値'],
            self._price_cols['安値']: self._sector_cols['安値']
        })
        merged_df['isSettlementDay'] = merged_df['isSettlementDay'].astype(bool).fillna(False)
        return merged_df
    
    def _calc_adjustment_factor(self, stock_price_with_shares: pd.DataFrame, stock_price: pd.DataFrame) -> pd.DataFrame:
        """株式分割・併合による発行済み株式数の変化を調整"""
        stock_price_to_adjust = self._extract_rows_to_adjust(stock_price_with_shares)
        stock_price_to_adjust = self._calc_shares_rate(stock_price_to_adjust)
        adjusted_stock_price = self._correct_shares_rate_for_non_adjustment(stock_price_to_adjust)
        stock_price = self._merge_shares_rate(stock_price, adjusted_stock_price)
        stock_price = self._handle_special_cases(stock_price)
        return self._calc_cumulative_shares_rate(stock_price)

    def _extract_rows_to_adjust(self, stock_price_with_shares_df: pd.DataFrame) -> pd.DataFrame:
        """株式分割・併合の対象行を抽出"""
        condition = (stock_price_with_shares_df[self._sector_cols['発行済み株式数']].notnull() | 
                    (stock_price_with_shares_df[self._price_cols['調整係数']] != 1))
        return stock_price_with_shares_df.loc[condition].copy()

    def _calc_shares_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """株式分割・併合による発行済み株式数の変化率を計算"""
        df[self._sector_cols['発行済み株式数']] = df.groupby(self._sector_cols['銘柄コード'])[self._sector_cols['発行済み株式数']].bfill()
        df['SharesRate'] = (df.groupby(self._sector_cols['銘柄コード'])[self._sector_cols['発行済み株式数']].shift(-1) / 
                           df[self._sector_cols['発行済み株式数']]).round(1)
        return df

    def _correct_shares_rate_for_non_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """株式分割・併合由来でない発行済み株式数変更の補正比率を1に修正"""
        shift_days = [1, 2, -1, -2]
        shift_columns = [f'Shift_AdjustmentFactor{i}' for i in shift_days]
        for shift_column, i in zip(shift_columns, shift_days):
            df[shift_column] = df.groupby(self._sector_cols['銘柄コード'])[self._price_cols['調整係数']].shift(i).fillna(1)
        df.loc[((df[shift_columns] == 1).all(axis=1) | (df['SharesRate'] == 1)), 'SharesRate'] = 1
        return df

    def _merge_shares_rate(self, stock_price: pd.DataFrame, df_to_calc_shares_rate: pd.DataFrame) -> pd.DataFrame:
        """株価調整用の発行済株式数比率を元の価格情報データフレームに結合"""
        df_to_calc_shares_rate = df_to_calc_shares_rate[df_to_calc_shares_rate['isSettlementDay']]
        df_to_calc_shares_rate['SharesRate'] = df_to_calc_shares_rate.groupby(self._sector_cols['銘柄コード'])['SharesRate'].shift(1)
        stock_price = pd.merge(
            stock_price,
            df_to_calc_shares_rate[[self._sector_cols['日付'], self._sector_cols['銘柄コード'], 
                                   self._sector_cols['発行済み株式数'], 'SharesRate']],
            how='left',
            on=[self._sector_cols['日付'], self._sector_cols['銘柄コード']]
        )
        stock_price['SharesRate'] = stock_price.groupby(self._sector_cols['銘柄コード'])['SharesRate'].shift(-1)
        stock_price['SharesRate'] = stock_price['SharesRate'].fillna(1)
        return stock_price

    def _handle_special_cases(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """最初の決算発表データ前の発行済株式数比率を手動で補正"""
        stock_price.loc[(stock_price[self._sector_cols['銘柄コード']] == '3064') & 
                       (stock_price[self._sector_cols['日付']] <= datetime(2013, 7, 25)), 'SharesRate'] = 1
        stock_price.loc[(stock_price[self._sector_cols['銘柄コード']] == '6920') & 
                       (stock_price[self._sector_cols['日付']] <= datetime(2013, 8, 9)), 'SharesRate'] = 1
        return stock_price

    def _calc_cumulative_shares_rate(self, stock_price: pd.DataFrame) -> pd.DataFrame:
        """発行済み株式数比率の累積積を計算"""
        stock_price = stock_price.sort_values(self._sector_cols['日付'], ascending=False)
        stock_price['CumulativeSharesRate'] = stock_price.groupby(self._sector_cols['銘柄コード'])['SharesRate'].cumprod()
        stock_price = stock_price.sort_values(self._sector_cols['日付'], ascending=True)
        stock_price['CumulativeSharesRate'] = stock_price['CumulativeSharesRate'].fillna(1)
        return stock_price

    def _adjust_shares(self, df: pd.DataFrame) -> pd.DataFrame:
        """発行済株式数を調整"""
        # 決算発表時以外が欠測値なので、後埋めする
        df[self._sector_cols['発行済み株式数']] = df.groupby(self._sector_cols['銘柄コード'], as_index=False)[self._sector_cols['発行済み株式数']].ffill() 
        # 初回決算発表以前の分を前埋め
        df[self._sector_cols['発行済み株式数']] = df.groupby(self._sector_cols['銘柄コード'], as_index=False)[self._sector_cols['発行済み株式数']].bfill() 
        df[self._sector_cols['発行済み株式数']] = df[self._sector_cols['発行済み株式数']] * df['CumulativeSharesRate']
        # 不要行の削除
        return df.drop(['SharesRate', 'CumulativeSharesRate'], axis=1)

    def _calc_marketcap(self, df: pd.DataFrame) -> pd.DataFrame:
        """時価総額を算出"""
        df[self._sector_cols['始値時価総額']] = df[self._sector_cols['始値']] * df[self._sector_cols['発行済み株式数']]
        df[self._sector_cols['終値時価総額']] = df[self._sector_cols['終値']] * df[self._sector_cols['発行済み株式数']]
        df[self._sector_cols['高値時価総額']] = df[self._sector_cols['高値']] * df[self._sector_cols['発行済み株式数']]
        df[self._sector_cols['安値時価総額']] = df[self._sector_cols['安値']] * df[self._sector_cols['発行済み株式数']]
        return df

    def _calc_correction_value(self, df: pd.DataFrame) -> pd.DataFrame:
        """指数算出補正値を算出"""
        df['OutstandingShares_forCorrection'] = \
            df.groupby(self._sector_cols['銘柄コード'])[self._sector_cols['発行済み株式数']].shift(1)
        df['OutstandingShares_forCorrection'] = df['OutstandingShares_forCorrection'].fillna(0)
        df['MarketCapClose_forCorrection'] = df[self._sector_cols['終値']] * df['OutstandingShares_forCorrection']
        df[self._sector_cols['指数算出用の補正値']] = df[self._sector_cols['終値時価総額']] - df['MarketCapClose_forCorrection']
        return df


if __name__ == '__main__':
    from acquisition.jquants_api_operations.facades import StockAcquisitionFacade
    
    # データ取得
    acq = StockAcquisitionFacade(filter="(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400')|(ScaleCategory=='TOPIX Small 1'))")
    stock_dfs = acq.get_stock_data_dict()

    # セクターインデックス計算（CSV定義）
    calculator = SectorIndex()
    sector_price_df, order_price_df = calculator.execute_full_calculation_from_csv(
        stock_dfs, 
        f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/topix1000.csv', 
        f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/TOPIX1000_price.parquet'
    )
    print(sector_price_df)

    # セクターインデックス計算（辞書定義）
    calculator2 = SectorIndex()
    calculator2.stock_dfs_dict = stock_dfs
    marketcap_df = calculator2.calculate_marketcap().stock_price_with_marketcap
    
    sector_index = calculator2.execute_calculation_from_dict(
        sector_stock_dict={
            'JPY+': ['2413', '3141', '4587', '1835', '4684'],
            'JPY-': ['7283', '7296', '5988', '8015', '7278']
        },
        stock_price_data=marketcap_df
    )
    
    print(sector_index)