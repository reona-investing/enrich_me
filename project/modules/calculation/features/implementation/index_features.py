from typing import Optional, Dict, Literal
import pandas as pd
from preprocessing import PreprocessingPipeline
from calculation.features.base import BaseFeatures
from utils.paths import Paths

class IndexFeatures(BaseFeatures):
    """インデックス系特徴量計算クラス"""
    
    def __init__(self):
        """初期化"""
        super().__init__()
        self.features_to_scrape_df = None
    
    def calculate_features(self, 
                          groups_setting: Dict = {},
                          names_setting: Dict = {},
                          currencies_type: Literal['relative', 'raw'] = 'relative',
                          commodity_type: Literal['JPY', 'raw'] = 'raw',
                          preprocessing_pipeline: Optional[PreprocessingPipeline] = None) -> pd.DataFrame:
        """
        インデックス系特徴量を計算し、self.features_dfを更新
        
        Args:
            groups_setting: 特徴量グループの採否設定
            names_setting: 特徴量の採否設定
            currencies_type: 通貨の処理方法
            commodity_type: コモディティの処理方法
            preprocessing_pipeline: 前処理パイプライン (任意)
            
        Returns:
            計算された特徴量データフレーム
        """
        # 特徴量選択
        self.features_to_scrape_df = self._select_features_to_scrape(
            groups_setting, names_setting
        )
        
        # インデックス特徴量計算
        self.features_df = self._calculate_indices_features(
            self.features_to_scrape_df, currencies_type, commodity_type
        )
        
        self.features_df = self.apply_preprocessing(preprocessing_pipeline)

        print('インデックス系特徴量の算出が完了しました。')
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
        print('インデックス系特徴量の前処理が完了しました。')
        return result
    
    def _select_features_to_scrape(self, groups_setting: Dict, names_setting: Dict) -> pd.DataFrame:
        """スクレイピング対象特徴量の選択"""
        features_to_scrape_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
        features_to_scrape_df['Path'] = (Paths.SCRAPED_DATA_FOLDER + '/' + 
                                       features_to_scrape_df['Group'] + '/' + 
                                       features_to_scrape_df['Path'])
        
        # グループ設定の適用
        if groups_setting:
            mapped = features_to_scrape_df['Group'].map(groups_setting)
            features_to_scrape_df['is_adopted'] = mapped.astype(bool).fillna(
                features_to_scrape_df['is_adopted']
            )
        
        # 個別名称設定の適用
        if names_setting:
            mapped = features_to_scrape_df['Name'].map(names_setting)
            features_to_scrape_df['is_adopted'] = mapped.astype(bool).fillna(
                features_to_scrape_df['is_adopted']
            )
        else:
            features_to_scrape_df['is_adopted'] = features_to_scrape_df['is_adopted'].replace(
                {'TRUE': True, 'FALSE': False}
            ).astype(bool)
        
        return features_to_scrape_df
    
    def _calculate_indices_features(self, 
                                  features_to_scrape_df: pd.DataFrame,
                                  currencies_type: str, 
                                  commodity_type: str) -> pd.DataFrame:
        """インデックス特徴量の計算メイン処理"""
        features_df = pd.DataFrame(columns=['Date'])
        
        for _, row in features_to_scrape_df.iterrows():
            should_convert_to_JPY = (row['Group'] == 'commodity') & (commodity_type == 'JPY')
            
            if should_convert_to_JPY:
                USDJPY_path = features_to_scrape_df.loc[
                    features_to_scrape_df['Name']=='USDJPY', 'Path'
                ].values[0]
                feature_df = self._calculate_1day_return_commodity_JPY(row, USDJPY_path)
            else:
                feature_df = self._calculate_1day_return(row)
            
            if feature_df is not None:
                features_df = pd.merge(features_df, feature_df, how='outer', on='Date').sort_values('Date')
        
        # データフレーム整形
        features_df = features_df.set_index('Date').ffill()
        
        # 通貨・債券特徴量の後処理
        features_df = self._post_process_features(features_df, features_to_scrape_df, currencies_type)
        
        print('インデックス系特徴量の算出が完了しました。')
        return features_df
    
    def _calculate_1day_return(self, row: pd.Series) -> Optional[pd.DataFrame]:
        """1日リターンの計算"""
        if not row['is_adopted']:
            return None
            
        raw_df = pd.read_parquet(row['Path'])
        feature_df = pd.DataFrame({
            'Date': pd.to_datetime(raw_df['Date'])
        })
        
        if row['Group'] == 'bond':
            feature_df[f'{row["Name"]}_1d_return'] = raw_df['Close'].diff(1)
        else:
            feature_df[f'{row["Name"]}_1d_return'] = raw_df['Close'].pct_change(1)
        
        return feature_df
    
    def _calculate_1day_return_commodity_JPY(self, row: pd.Series, USDJPY_path: str) -> Optional[pd.DataFrame]:
        """コモディティの円建て1日リターン計算"""
        if not row['is_adopted']:
            return None
            
        raw_df = pd.read_parquet(row['Path'])
        USDJPY_df = pd.read_parquet(USDJPY_path).rename(columns={'Close': 'USDJPYClose'})
        
        raw_df = pd.merge(raw_df, USDJPY_df[['Date', 'USDJPYClose']], on='Date', how='left')
        
        feature_df = pd.DataFrame({
            'Date': pd.to_datetime(raw_df['Date']),
            f'{row["Name"]}_1d_return': (raw_df['Close'] * raw_df['USDJPYClose']).pct_change(1)
        })
        
        return feature_df
    
    def _post_process_features(self, 
                             features_df: pd.DataFrame, 
                             features_to_scrape_df: pd.DataFrame,
                             currencies_type: str) -> pd.DataFrame:
        """通貨・債券特徴量の後処理"""
        # 通貨処理
        currency_is_adopted = features_to_scrape_df.loc[
            features_to_scrape_df['Group'] == 'currencies', 'is_adopted'
        ].all()
        
        if currency_is_adopted and currencies_type == 'relative':
            features_df = self._process_currency_relative_strength(features_df)
        
        # 債券処理
        bond_is_adopted = features_to_scrape_df.loc[
            features_to_scrape_df['Group'] == 'bond', 'is_adopted'
        ].any()
        
        if bond_is_adopted:
            features_df = self._process_bond_spreads(features_df)
        
        return features_df
    
    def _process_currency_relative_strength(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """通貨の相対強度計算"""
        for suffix in ['1d_return']:
            features_df[f'JPY_{suffix}'] = (
                -features_df[f'USDJPY_{suffix}'] - 
                features_df[f'EURJPY_{suffix}'] - 
                features_df[f'AUDJPY_{suffix}']
            ) / 3
            
            features_df[f'USD_{suffix}'] = (
                features_df[f'USDJPY_{suffix}'] - 
                features_df[f'EURUSD_{suffix}'] - 
                features_df[f'AUDUSD_{suffix}']
            ) / 3
            
            features_df[f'AUD_{suffix}'] = (
                features_df[f'AUDJPY_{suffix}'] + 
                features_df[f'AUDUSD_{suffix}'] - 
                features_df[f'EURAUD_{suffix}']
            ) / 3
            
            features_df[f'EUR_{suffix}'] = (
                features_df[f'EURJPY_{suffix}'] + 
                features_df[f'EURUSD_{suffix}'] + 
                features_df[f'EURAUD_{suffix}']
            ) / 3
            
            # 元の通貨ペア列を削除
            features_df.drop([
                f'USDJPY_{suffix}', f'EURJPY_{suffix}', f'AUDJPY_{suffix}',
                f'EURUSD_{suffix}', f'AUDUSD_{suffix}', f'EURAUD_{suffix}'
            ], axis=1, inplace=True)
        
        return features_df
    
    def _process_bond_spreads(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """債券スプレッドの計算"""
        bond_features = [col for col in features_df.columns if 'bond10' in col]
        
        for col in bond_features:
            spread_col = col.replace("10", "_diff")
            bond2_col = col.replace("10", "2")
            features_df[spread_col] = features_df[col] - features_df[bond2_col]
            features_df.drop([bond2_col], axis=1, inplace=True)
        
        return features_df