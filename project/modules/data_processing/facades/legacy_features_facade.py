"""
既存FeaturesCalculatorのファサード
新旧システムの並行運用と段階的移行を支援
"""
import pandas as pd
from typing import Dict, Any, Literal
import warnings

# 既存クラスのインポート
from calculation.features_calculator import FeaturesCalculator
from calculation.sector_index_calculator import SectorIndexCalculator

# 新システムのインポート
from data_processing.core.return_calculators import get_return_calculator
from data_processing.core.preprocessors import get_preprocessor
from data_processing.core.transformers.financial_transformers import (
    MomentumTransformer, VolatilityTransformer, RankingTransformer
)
from data_processing.core.pipeline import CalculationPipeline


class LegacyFeaturesFacade:
    """
    既存FeaturesCalculatorのファサード
    新旧システムの機能を統合し、段階的移行を支援
    """
    
    def __init__(self, use_new_system: bool = False, validate_consistency: bool = True):
        """
        Args:
            use_new_system: 新システムを使用するか
            validate_consistency: 新旧システムの結果の一貫性を検証するか
        """
        self.use_new_system = use_new_system
        self.validate_consistency = validate_consistency
        self._comparison_results = {}
    
    def calculate_features(self,
                          new_sector_price: pd.DataFrame,
                          new_sector_list: pd.DataFrame,
                          stock_dfs_dict: dict,
                          adopts_features_indices: bool = True,
                          adopts_features_price: bool = True,
                          groups_setting: dict = {},
                          names_setting: dict = {},
                          currencies_type: Literal['relative', 'raw'] = 'relative',
                          commodity_type: Literal['JPY', 'raw'] = 'raw',
                          adopt_1d_return: bool = True,
                          mom_duration: list = [5, 21],
                          vola_duration: list = [5, 21],
                          adopt_size_factor: bool = True,
                          adopt_eps_factor: bool = True,
                          adopt_sector_categorical: bool = True,
                          add_rank: bool = True) -> pd.DataFrame:
        """
        特徴量を算出（新旧システム対応）
        
        既存のFeaturesCalculator.calculate_featuresと同じインターフェース
        """
        
        if self.use_new_system:
            # 新システムで計算
            result = self._calculate_features_new_system(
                new_sector_price=new_sector_price,
                new_sector_list=new_sector_list,
                stock_dfs_dict=stock_dfs_dict,
                adopts_features_indices=adopts_features_indices,
                adopts_features_price=adopts_features_price,
                groups_setting=groups_setting,
                names_setting=names_setting,
                currencies_type=currencies_type,
                commodity_type=commodity_type,
                adopt_1d_return=adopt_1d_return,
                mom_duration=mom_duration,
                vola_duration=vola_duration,
                adopt_size_factor=adopt_size_factor,
                adopt_eps_factor=adopt_eps_factor,
                adopt_sector_categorical=adopt_sector_categorical,
                add_rank=add_rank
            )
        else:
            # 既存システムで計算
            result = FeaturesCalculator.calculate_features(
                new_sector_price=new_sector_price,
                new_sector_list=new_sector_list,
                stock_dfs_dict=stock_dfs_dict,
                adopts_features_indices=adopts_features_indices,
                adopts_features_price=adopts_features_price,
                groups_setting=groups_setting,
                names_setting=names_setting,
                currencies_type=currencies_type,
                commodity_type=commodity_type,
                adopt_1d_return=adopt_1d_return,
                mom_duration=mom_duration,
                vola_duration=vola_duration,
                adopt_size_factor=adopt_size_factor,
                adopt_eps_factor=adopt_eps_factor,
                adopt_sector_categorical=adopt_sector_categorical,
                add_rank=add_rank
            )
        
        # 一貫性検証（オプション）
        if self.validate_consistency and not self.use_new_system:
            self._validate_against_new_system(
                result,
                new_sector_price=new_sector_price,
                new_sector_list=new_sector_list,
                stock_dfs_dict=stock_dfs_dict,
                adopts_features_indices=adopts_features_indices,
                adopts_features_price=adopts_features_price,
                groups_setting=groups_setting,
                names_setting=names_setting,
                currencies_type=currencies_type,
                commodity_type=commodity_type,
                adopt_1d_return=adopt_1d_return,
                mom_duration=mom_duration,
                vola_duration=vola_duration,
                adopt_size_factor=adopt_size_factor,
                adopt_eps_factor=adopt_eps_factor,
                adopt_sector_categorical=adopt_sector_categorical,
                add_rank=add_rank
            )
        
        return result
    
    def _calculate_features_new_system(self, **kwargs) -> pd.DataFrame:
        """新システムでの特徴量算出"""
        
        # パイプラインを構築
        pipeline = CalculationPipeline()
        
        # 価格系特徴量の設定
        if kwargs.get('adopts_features_price', True):
            # 事前に1日リターンを計算
            sector_price_df = kwargs['new_sector_price'].copy()
            if '1d_return' not in sector_price_df.columns:
                sector_price_df['1d_return'] = sector_price_df.groupby(level='Sector')['Close'].pct_change(1)
            
            # モメンタム変換
            if kwargs.get('mom_duration'):
                momentum_transformer = MomentumTransformer(
                    windows=kwargs['mom_duration'],
                    base_column='1d_return',
                    add_rank=kwargs.get('add_rank', True)
                )
                pipeline.add_transformer(momentum_transformer)
            
            # ボラティリティ変換
            if kwargs.get('vola_duration'):
                volatility_transformer = VolatilityTransformer(
                    windows=kwargs['vola_duration'],
                    base_column='1d_return',
                    add_rank=kwargs.get('add_rank', True)
                )
                pipeline.add_transformer(volatility_transformer)
            
            # セクターカテゴリ変換
            if kwargs.get('adopt_sector_categorical', True):
                from data_processing.core.transformers import SectorCategoricalTransformer
                pipeline.add_transformer(SectorCategoricalTransformer())
        else:
            sector_price_df = kwargs['new_sector_price']
        
        # インデックス系特徴量は既存システムを使用（段階的移行）
        indices_df = pd.DataFrame()
        if kwargs.get('adopts_features_indices', True):
            try:
                features_to_scrape_df = FeaturesCalculator.select_features_to_scrape(
                    groups_setting=kwargs.get('groups_setting', {}),
                    names_setting=kwargs.get('names_setting', {})
                )
                indices_df = FeaturesCalculator.calculate_features_indices(
                    features_to_scrape_df=features_to_scrape_df,
                    currencies_type=kwargs.get('currencies_type', 'relative'),
                    commodity_type=kwargs.get('commodity_type', 'raw')
                )
            except Exception as e:
                warnings.warn(f"インデックス系特徴量の算出に失敗しました: {e}")
                indices_df = pd.DataFrame(index=sector_price_df.index)
        
        # 価格系特徴量を新システムで算出
        price_df = pd.DataFrame()
        if kwargs.get('adopts_features_price', True):
            try:
                # 新システムでパイプライン実行
                if len(pipeline.transformers) > 0:
                    price_df = pipeline.execute(sector_price_df)
                else:
                    price_df = sector_price_df.copy()
                
                # 既存システムの追加機能を補完
                price_df = self._add_legacy_price_features(
                    price_df,
                    kwargs['new_sector_price'],
                    kwargs['new_sector_list'],
                    kwargs['stock_dfs_dict'],
                    kwargs
                )
            except Exception as e:
                warnings.warn(f"価格系特徴量の算出で新システムが失敗、既存システムにフォールバック: {e}")
                price_df = FeaturesCalculator.calculate_features_price(
                    new_sector_price=kwargs['new_sector_price'],
                    new_sector_list=kwargs['new_sector_list'],
                    stock_dfs_dict=kwargs['stock_dfs_dict'],
                    adopt_1d_return=kwargs.get('adopt_1d_return', True),
                    mom_duration=kwargs.get('mom_duration', [5, 21]),
                    vola_duration=kwargs.get('vola_duration', [5, 21]),
                    adopt_size_factor=kwargs.get('adopt_size_factor', True),
                    adopt_eps_factor=kwargs.get('adopt_eps_factor', True),
                    adopt_sector_categorical=kwargs.get('adopt_sector_categorical', True),
                    add_rank=kwargs.get('add_rank', True)
                )
        
        # 結果を結合
        if not indices_df.empty and not price_df.empty:
            result = FeaturesCalculator.merge_features(indices_df, price_df)
        elif not indices_df.empty:
            result = indices_df
        elif not price_df.empty:
            result = price_df
        else:
            result = pd.DataFrame()
        
        return result.sort_index() if not result.empty else result
    
    def _add_legacy_price_features(self, base_df: pd.DataFrame,
                                 new_sector_price: pd.DataFrame,
                                 new_sector_list: pd.DataFrame,
                                 stock_dfs_dict: dict,
                                 kwargs: dict) -> pd.DataFrame:
        """既存システムの価格系特徴量を補完"""
        
        result = base_df.copy()
        
        # サイズファクター
        if kwargs.get('adopt_size_factor', True):
            try:
                new_sector_list['Code'] = new_sector_list['Code'].astype(str)
                stock_price_cap = SectorIndexCalculator.calc_marketcap(
                    stock_dfs_dict['price'], 
                    stock_dfs_dict['fin']
                )
                stock_price_cap = stock_price_cap[stock_price_cap['Code'].isin(new_sector_list['Code'])]
                stock_price_cap = pd.merge(
                    stock_price_cap, 
                    new_sector_list[['Code', 'Sector']], 
                    on='Code', 
                    how='left'
                )
                stock_price_cap = stock_price_cap[['Date', 'Code', 'Sector', 'MarketCapClose']]
                stock_price_cap = stock_price_cap.groupby(['Date', 'Sector'])[['MarketCapClose']].mean()
                
                result['MarketCapAtClose'] = stock_price_cap['MarketCapClose']
                
                if kwargs.get('add_rank', True):
                    result['MarketCap_rank'] = result['MarketCapAtClose'].groupby('Date').rank(ascending=False)
            except Exception as e:
                warnings.warn(f"サイズファクターの算出に失敗しました: {e}")
        
        # EPSファクター
        if kwargs.get('adopt_eps_factor', True):
            try:
                eps_df = stock_dfs_dict['fin'][['Code', 'Date', 'ForecastEPS']].copy()
                eps_df = pd.merge(
                    stock_dfs_dict['price'][['Date', 'Code']], 
                    eps_df, 
                    how='outer', 
                    on=['Date', 'Code']
                )
                eps_df = pd.merge(
                    new_sector_list[['Code', 'Sector']], 
                    eps_df, 
                    on='Code', 
                    how='right'
                )
                eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
                eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
                eps_df = pd.merge(
                    stock_dfs_dict['price'][['Date', 'Code']], 
                    eps_df, 
                    how='left', 
                    on=['Date', 'Code']
                )
                eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].ffill()
                eps_df['ForecastEPS'] = eps_df.groupby('Code')['ForecastEPS'].bfill()
                eps_df = eps_df.groupby(['Date', 'Sector'])[['ForecastEPS']].mean()
                eps_df['ForecastEPS_rank'] = eps_df.groupby('Date')['ForecastEPS'].rank(ascending=False)
                result[['ForecastEPS', 'ForecastEPS_rank']] = eps_df[['ForecastEPS', 'ForecastEPS_rank']].copy()
            except Exception as e:
                warnings.warn(f"EPSファクターの算出に失敗しました: {e}")
        
        # セクターカテゴリ変数
        if kwargs.get('adopt_sector_categorical', True):
            try:
                sector_replace_dict = {x: i for i, x in enumerate(result.index.get_level_values(1).unique())}
                result['Sector_cat'] = result.index.get_level_values(1)
                result['Sector_cat'] = result['Sector_cat'].replace(sector_replace_dict)
            except Exception as e:
                warnings.warn(f"セクターカテゴリ変数の作成に失敗しました: {e}")
        
        return result
    
    def _validate_against_new_system(self, legacy_result: pd.DataFrame, **kwargs) -> None:
        """既存システムの結果を新システムと比較検証"""
        try:
            # 新システム用のkwargsを準備（インデックス系を無効化して簡素化）
            new_kwargs = kwargs.copy()
            new_kwargs['adopts_features_indices'] = False  # 比較時はインデックス系を無効化
            
            new_result = self._calculate_features_new_system(**new_kwargs)
            
            # 共通カラムを比較
            common_columns = set(legacy_result.columns) & set(new_result.columns)
            comparison_results = {}
            
            for col in common_columns:
                legacy_series = legacy_result[col]
                new_series = new_result[col]
                
                # 共通インデックスで比較
                common_index = legacy_series.index.intersection(new_series.index)
                if len(common_index) > 0:
                    legacy_common = legacy_series.loc[common_index]
                    new_common = new_series.loc[common_index]
                    
                    # NaNを除外
                    valid_mask = legacy_common.notna() & new_common.notna()
                    if valid_mask.sum() > 0:
                        legacy_valid = legacy_common[valid_mask]
                        new_valid = new_common[valid_mask]
                        
                        # 統計的比較
                        correlation = legacy_valid.corr(new_valid)
                        max_diff = (legacy_valid - new_valid).abs().max()
                        mean_diff = (legacy_valid - new_valid).abs().mean()
                        
                        comparison_results[col] = {
                            'correlation': correlation,
                            'max_absolute_difference': max_diff,
                            'mean_absolute_difference': mean_diff,
                            'common_data_points': len(legacy_valid)
                        }
                        
                        # 大きな差異があれば警告
                        if correlation < 0.95 or max_diff > 0.01:
                            warnings.warn(
                                f"カラム '{col}' で新旧システム間に大きな差異があります。"
                                f"相関: {correlation:.4f}, 最大差: {max_diff:.6f}"
                            )
            
            self._comparison_results = comparison_results
            
        except Exception as e:
            warnings.warn(f"新旧システムの比較検証に失敗しました: {e}")
            self._comparison_results = {}
    
    def get_comparison_results(self) -> Dict[str, Any]:
        """新旧システムの比較結果を取得"""
        return self._comparison_results.copy()
    
    def calculate_features_with_pipeline_config(self,
                                              new_sector_price: pd.DataFrame,
                                              new_sector_list: pd.DataFrame,
                                              stock_dfs_dict: dict,
                                              config: Dict[str, Any]) -> pd.DataFrame:
        """
        設定ファイルベースでの特徴量算出（新システム機能）
        
        Args:
            new_sector_price: セクター価格データ
            new_sector_list: セクター定義
            stock_dfs_dict: 個別株価データ
            config: パイプライン設定
            
        Returns:
            特徴量データフレーム
        """
        if not self.use_new_system:
            raise ValueError("設定ファイルベースの機能は新システムでのみ利用可能です")
        
        # 設定からパイプラインを構築
        pipeline = self._build_pipeline_from_config(config)
        
        # インデックス系特徴量（既存システムを使用）
        indices_df = pd.DataFrame()
        if config.get('include_indices_features', True):
            features_to_scrape_df = FeaturesCalculator.select_features_to_scrape(
                groups_setting=config.get('groups_setting', {}),
                names_setting=config.get('names_setting', {})
            )
            indices_df = FeaturesCalculator.calculate_features_indices(
                features_to_scrape_df=features_to_scrape_df,
                currencies_type=config.get('currencies_type', 'relative'),
                commodity_type=config.get('commodity_type', 'raw')
            )
        
        # 価格系特徴量（新システム）
        price_df = pipeline.execute(new_sector_price)
        
        # 既存システムの追加機能を補完
        if config.get('include_legacy_features', True):
            price_df = self._add_legacy_price_features(
                price_df, new_sector_price, new_sector_list, stock_dfs_dict, config
            )
        
        # 結果を結合
        if not indices_df.empty and not price_df.empty:
            result = FeaturesCalculator.merge_features(indices_df, price_df)
        elif not indices_df.empty:
            result = indices_df
        else:
            result = price_df
        
        return result.sort_index()
    
    def _build_pipeline_from_config(self, config: Dict[str, Any]) -> CalculationPipeline:
        """設定からパイプラインを構築"""
        pipeline = CalculationPipeline()
        
        # リターン算出器の設定
        if 'return_calculator' in config:
            calc_config = config['return_calculator']
            calculator = get_return_calculator(**calc_config)
            pipeline.set_return_calculator(calculator)
        
        # 前処理器の設定
        if 'preprocessor' in config:
            prep_config = config['preprocessor']
            preprocessor = get_preprocessor(**prep_config)
            pipeline.set_preprocessor(preprocessor)
        
        # 変換器の設定
        if 'transformers' in config:
            for trans_config in config['transformers']:
                transformer = self._create_transformer_from_config(trans_config)
                pipeline.add_transformer(transformer)
        
        return pipeline
    
    def _create_transformer_from_config(self, config: Dict[str, Any]):
        """設定から変換器を作成"""
        transformer_type = config.get('type')
        
        if transformer_type == 'momentum':
            return MomentumTransformer(**{k: v for k, v in config.items() if k != 'type'})
        elif transformer_type == 'volatility':
            return VolatilityTransformer(**{k: v for k, v in config.items() if k != 'type'})
        elif transformer_type == 'ranking':
            return RankingTransformer(**{k: v for k, v in config.items() if k != 'type'})
        else:
            raise ValueError(f"未対応の変換器タイプです: {transformer_type}")
    
    def migrate_to_new_system(self, validation_threshold: float = 0.95) -> bool:
        """
        新システムへの移行判定
        
        Args:
            validation_threshold: 相関閾値（これ以上なら移行OK）
            
        Returns:
            移行可能かどうか
        """
        if not self._comparison_results:
            warnings.warn("比較結果がありません。先にvalidate_consistencyを有効にして計算を実行してください。")
            return False
        
        # 全カラムの相関が閾値以上かチェック
        all_correlations = [
            result['correlation'] 
            for result in self._comparison_results.values() 
            if not pd.isna(result['correlation'])
        ]
        
        if not all_correlations:
            return False
        
        min_correlation = min(all_correlations)
        avg_correlation = sum(all_correlations) / len(all_correlations)
        
        print(f"新旧システム比較結果:")
        print(f"  最小相関: {min_correlation:.4f}")
        print(f"  平均相関: {avg_correlation:.4f}")
        print(f"  検証カラム数: {len(all_correlations)}")
        
        migration_ready = min_correlation >= validation_threshold
        
        if migration_ready:
            print(f"✓ 新システムへの移行準備が完了しました（閾値: {validation_threshold}）")
        else:
            print(f"✗ 新システムへの移行にはさらなる調整が必要です（閾値: {validation_threshold}）")
        
        return migration_ready
    
    def get_feature_coverage_report(self) -> Dict[str, Any]:
        """特徴量カバレッジレポートを取得"""
        if not self._comparison_results:
            return {"error": "比較結果がありません"}
        
        total_features = len(self._comparison_results)
        high_correlation_features = sum(
            1 for result in self._comparison_results.values() 
            if not pd.isna(result['correlation']) and result['correlation'] >= 0.95
        )
        
        return {
            "total_features_compared": total_features,
            "high_correlation_features": high_correlation_features,
            "coverage_ratio": high_correlation_features / total_features if total_features > 0 else 0,
            "detailed_results": self._comparison_results
        }


# 便利関数
def create_legacy_facade(use_new_system: bool = False, 
                        validate_consistency: bool = True) -> LegacyFeaturesFacade:
    """LegacyFeaturesFacadeのファクトリー関数"""
    return LegacyFeaturesFacade(
        use_new_system=use_new_system,
        validate_consistency=validate_consistency
    )


def compare_legacy_vs_new_system(new_sector_price: pd.DataFrame,
                               new_sector_list: pd.DataFrame,
                               stock_dfs_dict: dict,
                               **kwargs) -> Dict[str, Any]:
    """
    新旧システムの直接比較
    
    Returns:
        比較結果の詳細レポート
    """
    facade = LegacyFeaturesFacade(use_new_system=False, validate_consistency=True)
    
    # 既存システムで計算（内部で新システムとの比較も実行）
    legacy_result = facade.calculate_features(
        new_sector_price=new_sector_price,
        new_sector_list=new_sector_list,
        stock_dfs_dict=stock_dfs_dict,
        **kwargs
    )
    
    # 新システムで計算
    facade_new = LegacyFeaturesFacade(use_new_system=True, validate_consistency=False)
    new_result = facade_new.calculate_features(
        new_sector_price=new_sector_price,
        new_sector_list=new_sector_list,
        stock_dfs_dict=stock_dfs_dict,
        **kwargs
    )
    
    return {
        "legacy_result": legacy_result,
        "new_result": new_result,
        "comparison_results": facade.get_comparison_results(),
        "coverage_report": facade.get_feature_coverage_report()
    }