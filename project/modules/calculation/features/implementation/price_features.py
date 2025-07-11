from typing import Optional, Dict, List
import pandas as pd
from timeseries_data.preprocessing import PreprocessingPipeline
from calculation.features.base import BaseFeatures


class PriceFeatures(BaseFeatures):
    """価格系特徴量計算クラス"""
    
    def __init__(self):
        """初期化"""
        super().__init__()
    
    def calculate_features(self,
                          new_sector_price: pd.DataFrame,
                          new_sector_list: pd.DataFrame,
                          stock_dfs_dict: Dict,
                          adopt_1d_return: bool = True,
                          mom_duration: List[int] = [5, 21], 
                          vola_duration: List[int] = [5, 21],
                          adopt_size_factor: bool = True,
                          adopt_eps_factor: bool = True,
                          adopt_sector_categorical: bool = True,
                          add_rank: bool = True,
                          preprocessing_pipeline: Optional[PreprocessingPipeline] = None) -> pd.DataFrame:
        """
        価格系特徴量を計算し、self.features_dfを更新
        
        Args:
            new_sector_price: セクター価格データ
            new_sector_list: セクターリストデータ
            stock_dfs_dict: 株価データ辞書
            adopt_1d_return: 1日リターンを採用するか
            mom_duration: モメンタム計算期間
            vola_duration: ボラティリティ計算期間
            adopt_size_factor: サイズファクターを採用するか
            adopt_eps_factor: EPSファクターを採用するか
            adopt_sector_categorical: セクターカテゴリを採用するか
            add_rank: ランキングを追加するか
            preprocessing_pipeline: 前処理パイプライン (任意)
            
        Returns:
            計算された特徴量データフレーム
        """
        # self.features_dfを初期化
        self.features_df = pd.DataFrame()
        
        # 基本リターン特徴量
        if adopt_1d_return:
            self._add_return_features(new_sector_price, add_rank)
        
        # モメンタム特徴量
        if mom_duration:
            self._add_momentum_features(mom_duration, add_rank)
        
        # ボラティリティ特徴量
        if vola_duration:
            self._add_volatility_features(vola_duration, add_rank)
        
        # サイズファクター
        if adopt_size_factor:
            self._add_size_factor(new_sector_list, stock_dfs_dict, add_rank)
        
        # EPSファクター
        if adopt_eps_factor:
            self._add_eps_factor(new_sector_list, stock_dfs_dict, add_rank)
        
        # セクターカテゴリ
        if adopt_sector_categorical:
            self._add_sector_categorical()
        
        self.features_df = self.apply_preprocessing(preprocessing_pipeline)
        
        print('価格系特徴量の算出が完了しました。')
        return self.features_df.copy()
    
    def apply_preprocessing(self, pipeline: Optional[PreprocessingPipeline] = None) -> pd.DataFrame:
        """
        前処理パイプラインを適用し、self.features_dfを更新
        
        Args:
            pipeline: 前処理パイプライン
            
        Returns:
            前処理後の特徴量データフレーム
        """
        result = super().apply_preprocessing(pipeline)
        print('価格系特徴量の前処理が完了しました。')
        return result
    
    def _add_return_features(self, new_sector_price: pd.DataFrame, add_rank: bool) -> None:
        """リターン特徴量をself.features_dfに追加"""
        self.features_df['1d_return'] = new_sector_price['1d_return']
        if add_rank:
            self.features_df['1d_return_rank'] = self.features_df['1d_return'].groupby('Date').rank(ascending=False)
    
    def _add_momentum_features(self, mom_duration: List[int], add_rank: bool) -> None:
        """モメンタム特徴量をself.features_dfに追加"""
        assert '1d_return' in self.features_df.columns, "モメンタム計算には1d_returnが必要です"
        
        days_to_exclude = 1
        for n in mom_duration:
            col_name = f'{n}d_mom'
            self.features_df[col_name] = (
                self.features_df['1d_return']
                .groupby('Sector')
                .rolling(n - days_to_exclude)
                .mean()
                .reset_index(0, drop=True)
            )
            self.features_df[col_name] = self.features_df[col_name].groupby('Sector').shift(days_to_exclude)
            
            if add_rank:
                self.features_df[f'{col_name}_rank'] = self.features_df[col_name].groupby('Date').rank(ascending=False)
            
            days_to_exclude = n
    
    def _add_volatility_features(self, vola_duration: List[int], add_rank: bool) -> None:
        """ボラティリティ特徴量をself.features_dfに追加"""
        assert '1d_return' in self.features_df.columns, "ボラティリティ計算には1d_returnが必要です"
        
        days_to_exclude = 1
        for n in vola_duration:
            col_name = f'{n}d_vola'
            self.features_df[col_name] = (
                self.features_df['1d_return']
                .groupby('Sector')
                .rolling(n - days_to_exclude)
                .std()
                .reset_index(0, drop=True)
            )
            self.features_df[col_name] = self.features_df[col_name].groupby('Sector').shift(days_to_exclude)
            
            if add_rank:
                self.features_df[f'{col_name}_rank'] = self.features_df[col_name].groupby('Date').rank(ascending=False)
            
            days_to_exclude = n
    
    def _add_size_factor(self, new_sector_list: pd.DataFrame, stock_dfs_dict: Dict, add_rank: bool) -> None:
        """サイズファクターをself.features_dfに追加"""
        from calculation.sector_index.sector_index import SectorIndex

        new_sector_list['Code'] = new_sector_list['Code'].astype(str)
        sic = SectorIndex()
        stock_price_cap = sic.calc_marketcap(
            stock_dfs_dict['price'], stock_dfs_dict['fin']
        )
        
        stock_price_cap = stock_price_cap[stock_price_cap['Code'].isin(new_sector_list['Code'])]
        stock_price_cap = pd.merge(
            stock_price_cap, new_sector_list[['Code', 'Sector']], on='Code', how='left'
        )
        
        sector_marketcap = (
            stock_price_cap[['Date', 'Code', 'Sector', 'MarketCapClose']]
            .groupby(['Date', 'Sector'])[['MarketCapClose']]
            .mean()
        )
        
        self.features_df['MarketCapAtClose'] = sector_marketcap['MarketCapClose']
        
        if add_rank:
            self.features_df['MarketCap_rank'] = self.features_df['MarketCapAtClose'].groupby('Date').rank(ascending=False)
    
    def _add_eps_factor(self, new_sector_list: pd.DataFrame, stock_dfs_dict: Dict, add_rank: bool) -> None:
        """EPSファクターをself.features_dfに追加"""
        eps_df = stock_dfs_dict['fin'][['Code', 'Date', 'ForecastEPS']].copy()
        eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='outer', on=['Date', 'Code'])
        eps_df = pd.merge(new_sector_list[['Code', 'Sector']], eps_df, on='Code', how='right')
        
        # EPS前処理
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill().bfill()
        eps_df = pd.merge(stock_dfs_dict['price'][['Date', 'Code']], eps_df, how='left', on=['Date', 'Code'])
        eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill().bfill()
        
        # セクター平均EPS
        sector_eps = eps_df.groupby(['Date', 'Sector'])[['ForecastEPS']].mean()
        
        if add_rank:
            sector_eps['ForecastEPS_rank'] = sector_eps.groupby('Date')['ForecastEPS'].rank(ascending=False)
            self.features_df[['ForecastEPS', 'ForecastEPS_rank']] = sector_eps[['ForecastEPS', 'ForecastEPS_rank']]
        else:
            self.features_df['ForecastEPS'] = sector_eps['ForecastEPS']
    
    def _add_sector_categorical(self) -> None:
        """セクターカテゴリ変数をself.features_dfに追加"""
        sector_replace_dict = {
            sector: i for i, sector in enumerate(self.features_df.index.get_level_values(1).unique())
        }
        self.features_df['Sector_cat'] = (
            self.features_df.index.get_level_values(1)
            .map(sector_replace_dict)
            .astype('category')
        )