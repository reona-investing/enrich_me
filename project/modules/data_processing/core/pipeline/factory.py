"""
パイプラインファクトリー - 設定からパイプラインを自動生成
"""
from typing import Dict, Any, List, Optional
from datetime import datetime
import warnings

from data_processing.core.pipeline.core_pipeline import CalculationPipeline, BatchPipeline
from data_processing.core.return_calculators import get_return_calculator
from data_processing.core.preprocessors import get_preprocessor
from data_processing.core.transformers.financial_transformers import get_financial_transformer
from data_processing.core.transformers.statistical_transformers import get_statistical_transformer
from data_processing.core.data_generators.implementations import get_data_generator
from data_processing.config.pipeline_configs import PipelineConfigManager, ConfigValidator


class PipelineFactory:
    """設定からパイプラインを自動生成するファクトリークラス"""
    
    def __init__(self):
        self.config_manager = PipelineConfigManager()
        self.validator = ConfigValidator()
    
    def create_pipeline_from_config(self, config: Dict[str, Any], 
                                  validate_config: bool = True) -> CalculationPipeline:
        """
        設定辞書からパイプラインを生成
        
        Args:
            config: パイプライン設定辞書
            validate_config: 設定の検証を行うか
            
        Returns:
            設定済みパイプライン
        """
        if validate_config:
            validation_result = self.validator.validate_config(config)
            if validation_result["errors"]:
                raise ValueError(f"設定エラー: {validation_result['errors']}")
            if validation_result["warnings"]:
                for warning in validation_result["warnings"]:
                    warnings.warn(warning)
        
        pipeline = CalculationPipeline()
        
        # データ生成器の設定
        if "data_generator" in config:
            data_generator = self._create_data_generator(config["data_generator"])
            # データ生成器はパイプラインに直接組み込まず、事前処理として使用
        
        # リターン算出器の設定
        if "return_calculator" in config:
            return_calculator = self._create_return_calculator(config["return_calculator"])
            pipeline.set_return_calculator(return_calculator)
        
        # 前処理器の設定
        if "preprocessor" in config:
            preprocessor = self._create_preprocessor(config["preprocessor"])
            pipeline.set_preprocessor(preprocessor)
        
        # 変換器の設定
        if "transformers" in config:
            for transformer_config in config["transformers"]:
                transformer = self._create_transformer(transformer_config)
                pipeline.add_transformer(transformer)
        
        return pipeline
    
    def create_pipeline_from_file(self, config_file: str) -> CalculationPipeline:
        """
        設定ファイルからパイプラインを生成
        
        Args:
            config_file: 設定ファイル名（拡張子なし）
            
        Returns:
            設定済みパイプライン
        """
        config = self.config_manager.load_config(config_file)
        return self.create_pipeline_from_config(config)
    
    def create_batch_pipeline_from_config(self, config: Dict[str, Any]) -> BatchPipeline:
        """
        設定からバッチパイプラインを生成
        
        Args:
            config: バッチパイプライン設定辞書
            
        Returns:
            バッチパイプライン
        """
        # ベースパイプラインを作成
        base_pipeline = self.create_pipeline_from_config(config.get("base_pipeline", {}))
        
        # バッチパイプラインを作成
        batch_pipeline = BatchPipeline(base_pipeline)
        
        return batch_pipeline
    
    def _create_return_calculator(self, calc_config: Dict[str, Any]):
        """リターン算出器を作成"""
        calc_type = calc_config.get("type", "daily")
        
        # タイプ別のパラメータマッピング
        params = calc_config.copy()
        params.pop("type", None)
        
        return get_return_calculator(calc_type, **params)
    
    def _create_preprocessor(self, prep_config: Dict[str, Any]):
        """前処理器を作成"""
        prep_type = prep_config.get("type", "none")
        
        # タイプ別のパラメータマッピング
        params = prep_config.copy()
        params.pop("type", None)
        
        # 日付文字列をdatetimeオブジェクトに変換
        if "train_start" in params and isinstance(params["train_start"], str):
            params["train_start"] = datetime.fromisoformat(params["train_start"])
        if "train_end" in params and isinstance(params["train_end"], str):
            params["train_end"] = datetime.fromisoformat(params["train_end"])
        
        return get_preprocessor(prep_type, **params)
    
    def _create_transformer(self, trans_config: Dict[str, Any]):
        """変換器を作成"""
        trans_type = trans_config.get("type")
        
        if not trans_type:
            raise ValueError("変換器のtypeが指定されていません")
        
        # タイプ別のパラメータマッピング
        params = trans_config.copy()
        params.pop("type", None)
        
        # 金融変換器を試行
        try:
            return get_financial_transformer(trans_type, **params)
        except ValueError:
            pass
        
        # 統計変換器を試行
        try:
            return get_statistical_transformer(trans_type, **params)
        except ValueError:
            pass
        
        raise ValueError(f"未対応の変換器タイプです: {trans_type}")
    
    def _create_data_generator(self, gen_config: Dict[str, Any]):
        """データ生成器を作成"""
        gen_type = gen_config.get("type", "sector_index")
        
        # タイプ別のパラメータマッピング
        params = gen_config.copy()
        params.pop("type", None)
        
        return get_data_generator(gen_type, **params)


class PresetPipelineFactory:
    """プリセットパイプラインのファクトリー"""
    
    @staticmethod
    def create_basic_features_pipeline() -> CalculationPipeline:
        """基本的な特徴量パイプライン"""
        from ..transformers.financial_transformers import MomentumTransformer, VolatilityTransformer
        
        return (
            CalculationPipeline()
            .add_transformer(MomentumTransformer(windows=[5, 21], add_rank=True))
            .add_transformer(VolatilityTransformer(windows=[5, 21], add_rank=True))
        )
    
    @staticmethod
    def create_pca_features_pipeline(train_start: datetime, train_end: datetime) -> CalculationPipeline:
        """PCA前処理付き特徴量パイプライン"""
        from ..transformers.financial_transformers import MomentumTransformer
        
        return (
            CalculationPipeline()
            .set_preprocessor(get_preprocessor(
                'pca_market_factor_removal',
                n_components=1,
                train_start=train_start,
                train_end=train_end
            ))
            .add_transformer(MomentumTransformer(windows=[5, 21], base_column='Target'))
        )
    
    @staticmethod
    def create_technical_analysis_pipeline() -> CalculationPipeline:
        """テクニカル分析パイプライン"""
        from ..transformers.financial_transformers import TechnicalIndicatorTransformer, MomentumTransformer, VolatilityTransformer
        
        return (
            CalculationPipeline()
            .add_transformer(TechnicalIndicatorTransformer(
                indicators=['rsi', 'macd', 'bb'],
                price_columns=['Close']
            ))
            .add_transformer(MomentumTransformer(windows=[5, 10, 21]))
            .add_transformer(VolatilityTransformer(windows=[5, 10, 21]))
        )
    
    @staticmethod
    def create_statistical_analysis_pipeline() -> CalculationPipeline:
        """統計分析パイプライン"""
        from ..transformers.statistical_transformers import (
            StatisticalMomentsTransformer, ZScoreTransformer, QuantileTransformer
        )
        
        return (
            CalculationPipeline()
            .add_transformer(StatisticalMomentsTransformer(
                windows=[21, 60],
                moments=['mean', 'std', 'skew']
            ))
            .add_transformer(ZScoreTransformer(windows=[21, 60]))
            .add_transformer(QuantileTransformer(
                windows=[21, 60],
                quantiles=[0.1, 0.25, 0.75, 0.9]
            ))
        )
    
    @staticmethod
    def create_multi_timeframe_pipeline() -> CalculationPipeline:
        """マルチタイムフレーム分析パイプライン"""
        from ..transformers.financial_transformers import MomentumTransformer, VolatilityTransformer, RankingTransformer
        
        return (
            CalculationPipeline()
            .add_transformer(MomentumTransformer(windows=[3, 5, 10, 21, 60]))
            .add_transformer(VolatilityTransformer(windows=[3, 5, 10, 21, 60]))
            .add_transformer(RankingTransformer())
        )
    
    @staticmethod
    def create_currency_analysis_pipeline() -> CalculationPipeline:
        """通貨分析パイプライン"""
        from ..transformers.financial_transformers import CurrencyRelativeStrengthTransformer, MomentumTransformer
        
        return (
            CalculationPipeline()
            .add_transformer(CurrencyRelativeStrengthTransformer())
            .add_transformer(MomentumTransformer(
                windows=[5, 21],
                base_column='JPY_1d_return'
            ))
        )


class ConfigBasedPipelineManager:
    """設定ベースのパイプライン管理クラス"""
    
    def __init__(self, config_directory: Optional[str] = None):
        self.factory = PipelineFactory()
        self.config_manager = PipelineConfigManager(config_directory) if config_directory else PipelineConfigManager()
        self.pipeline_cache = {}
    
    def get_pipeline(self, config_name: str, use_cache: bool = True) -> CalculationPipeline:
        """
        設定名からパイプラインを取得（キャッシュ機能付き）
        
        Args:
            config_name: 設定名
            use_cache: キャッシュを使用するか
            
        Returns:
            パイプライン
        """
        if use_cache and config_name in self.pipeline_cache:
            return self.pipeline_cache[config_name].clone()
        
        pipeline = self.factory.create_pipeline_from_file(config_name)
        
        if use_cache:
            self.pipeline_cache[config_name] = pipeline.clone()
        
        return pipeline
    
    def create_custom_pipeline(self, config_name: str, config: Dict[str, Any], 
                             save_config: bool = True) -> CalculationPipeline:
        """
        カスタム設定からパイプラインを作成
        
        Args:
            config_name: 設定名
            config: パイプライン設定
            save_config: 設定を保存するか
            
        Returns:
            パイプライン
        """
        if save_config:
            self.config_manager.save_config(config_name, config)
        
        pipeline = self.factory.create_pipeline_from_config(config)
        self.pipeline_cache[config_name] = pipeline.clone()
        
        return pipeline
    
    def list_available_pipelines(self) -> List[str]:
        """利用可能なパイプライン設定一覧を取得"""
        configs = self.config_manager.list_configs()
        return [config["name"] for config in configs]
    
    def get_pipeline_info(self, config_name: str) -> Dict[str, Any]:
        """パイプライン情報を取得"""
        try:
            config = self.config_manager.load_config(config_name)
            pipeline = self.get_pipeline(config_name)
            
            return {
                "config_name": config_name,
                "description": config.get("description", ""),
                "pipeline_summary": pipeline.get_configuration_summary(),
                "config": config
            }
        except Exception as e:
            return {
                "config_name": config_name,
                "error": str(e)
            }
    
    def clear_cache(self) -> None:
        """パイプラインキャッシュをクリア"""
        self.pipeline_cache.clear()


# 便利関数
def create_pipeline_from_config(config: Dict[str, Any]) -> CalculationPipeline:
    """設定からパイプラインを作成する便利関数"""
    factory = PipelineFactory()
    return factory.create_pipeline_from_config(config)


def create_pipeline_from_file(config_file: str) -> CalculationPipeline:
    """設定ファイルからパイプラインを作成する便利関数"""
    factory = PipelineFactory()
    return factory.create_pipeline_from_file(config_file)


def create_preset_pipeline(preset_type: str, **kwargs) -> CalculationPipeline:
    """プリセットパイプラインを作成する便利関数"""
    if preset_type == "basic_features":
        return PresetPipelineFactory.create_basic_features_pipeline()
    elif preset_type == "pca_features":
        train_start = kwargs.get('train_start', datetime(2020, 1, 1))
        train_end = kwargs.get('train_end', datetime(2022, 12, 31))
        return PresetPipelineFactory.create_pca_features_pipeline(train_start, train_end)
    elif preset_type == "technical_analysis":
        return PresetPipelineFactory.create_technical_analysis_pipeline()
    elif preset_type == "statistical_analysis":
        return PresetPipelineFactory.create_statistical_analysis_pipeline()
    elif preset_type == "multi_timeframe":
        return PresetPipelineFactory.create_multi_timeframe_pipeline()
    elif preset_type == "currency_analysis":
        return PresetPipelineFactory.create_currency_analysis_pipeline()
    else:
        raise ValueError(f"未対応のプリセットタイプ: {preset_type}")


def get_pipeline_manager(config_directory: Optional[str] = None) -> ConfigBasedPipelineManager:
    """パイプライン管理者を取得する便利関数"""
    return ConfigBasedPipelineManager(config_directory)


# パイプライン設定テンプレート関数
def create_research_pipeline_config(research_type: str, **kwargs) -> Dict[str, Any]:
    """研究タイプに応じたパイプライン設定を作成"""
    
    if research_type == "factor_research":
        return {
            "pipeline_name": f"factor_research_{kwargs.get('factor_name', 'custom')}",
            "description": f"ファクター研究: {kwargs.get('factor_name', 'カスタム')}",
            "data_generator": {
                "type": "custom_factor",
                "factor_type": kwargs.get("factor_type", "custom"),
                "factor_definition": kwargs.get("factor_definition", {})
            },
            "return_calculator": {
                "type": kwargs.get("return_type", "intraday")
            },
            "preprocessor": {
                "type": "pca_market_factor_removal",
                "n_components": kwargs.get("pca_components", 1),
                "train_start": kwargs.get("train_start", "2020-01-01"),
                "train_end": kwargs.get("train_end", "2022-12-31")
            },
            "transformers": [
                {
                    "type": "momentum",
                    "windows": kwargs.get("momentum_windows", [5, 21]),
                    "base_column": "Target",
                    "add_rank": True
                },
                {
                    "type": "volatility",
                    "windows": kwargs.get("volatility_windows", [5, 21]),
                    "base_column": "Target",
                    "add_rank": True
                }
            ]
        }
    
    elif research_type == "sector_rotation":
        return {
            "pipeline_name": "sector_rotation_analysis",
            "description": "セクターローテーション分析",
            "return_calculator": {
                "type": "daily"
            },
            "transformers": [
                {
                    "type": "momentum",
                    "windows": [5, 10, 21, 60],
                    "base_column": "1d_return",
                    "add_rank": True
                },
                {
                    "type": "volatility",
                    "windows": [5, 10, 21, 60],
                    "base_column": "1d_return",
                    "add_rank": True
                },
                {
                    "type": "statistical_moments",
                    "windows": [21, 60],
                    "moments": ["mean", "std", "skew"],
                    "base_column": "1d_return"
                },
                {
                    "type": "ranking",
                    "target_columns": kwargs.get("ranking_columns", ["5d_mom", "21d_mom"])
                }
            ]
        }
    
    elif research_type == "risk_analysis":
        return {
            "pipeline_name": "risk_analysis",
            "description": "リスク分析パイプライン",
            "return_calculator": {
                "type": "daily"
            },
            "preprocessor": {
                "type": "outlier_removal",
                "method": "iqr",
                "threshold": 1.5
            },
            "transformers": [
                {
                    "type": "volatility",
                    "windows": [5, 10, 21, 60],
                    "base_column": "1d_return",
                    "add_rank": True
                },
                {
                    "type": "outlier_detection",
                    "windows": [21, 60],
                    "methods": ["iqr", "zscore"],
                    "target_columns": ["1d_return"]
                },
                {
                    "type": "percentile_rank",
                    "windows": [21, 60],
                    "target_columns": ["1d_return"]
                },
                {
                    "type": "rolling_beta",
                    "benchmark_column": kwargs.get("benchmark_column", "market_return"),
                    "windows": [21, 60]
                }
            ]
        }
    
    elif research_type == "technical_momentum":
        return {
            "pipeline_name": "technical_momentum_analysis",
            "description": "テクニカル・モメンタム分析",
            "transformers": [
                {
                    "type": "technical_indicators",
                    "indicators": ["rsi", "macd", "bb"],
                    "price_columns": ["Close"]
                },
                {
                    "type": "momentum",
                    "windows": [5, 10, 21, 60],
                    "base_column": "1d_return",
                    "add_rank": True
                },
                {
                    "type": "trend_analysis",
                    "windows": [21, 60],
                    "trend_methods": ["linear_slope", "trend_strength"],
                    "base_column": "Close"
                }
            ]
        }
    
    else:
        raise ValueError(f"未対応の研究タイプ: {research_type}")


def create_multi_strategy_config(strategies: List[str], **kwargs) -> Dict[str, Any]:
    """複数戦略を組み合わせた設定を作成"""
    transformers = []
    
    # 基本的なリターン・モメンタム・ボラティリティ
    if "momentum" in strategies:
        transformers.append({
            "type": "momentum",
            "windows": kwargs.get("momentum_windows", [5, 21]),
            "base_column": kwargs.get("base_column", "1d_return"),
            "add_rank": True
        })
    
    if "volatility" in strategies:
        transformers.append({
            "type": "volatility", 
            "windows": kwargs.get("volatility_windows", [5, 21]),
            "base_column": kwargs.get("base_column", "1d_return"),
            "add_rank": True
        })
    
    # テクニカル分析
    if "technical" in strategies:
        transformers.append({
            "type": "technical_indicators",
            "indicators": kwargs.get("technical_indicators", ["rsi", "macd"]),
            "price_columns": ["Close"]
        })
    
    # 統計分析
    if "statistical" in strategies:
        transformers.extend([
            {
                "type": "statistical_moments",
                "windows": [21, 60],
                "moments": ["mean", "std", "skew"],
                "base_column": kwargs.get("base_column", "1d_return")
            },
            {
                "type": "zscore",
                "windows": [21, 60],
                "target_columns": [kwargs.get("base_column", "1d_return")]
            }
        ])
    
    # 通貨分析
    if "currency" in strategies:
        transformers.append({
            "type": "currency_relative_strength",
            "return_suffix": "_1d_return"
        })
    
    # ランキング
    if "ranking" in strategies:
        transformers.append({
            "type": "ranking"
        })
    
    config = {
        "pipeline_name": f"multi_strategy_{'_'.join(strategies)}",
        "description": f"マルチ戦略分析: {', '.join(strategies)}",
        "transformers": transformers
    }
    
    # 前処理の設定
    if kwargs.get("use_pca", False):
        config["preprocessor"] = {
            "type": "pca_market_factor_removal",
            "n_components": kwargs.get("pca_components", 1),
            "train_start": kwargs.get("train_start", "2020-01-01"),
            "train_end": kwargs.get("train_end", "2022-12-31")
        }
    elif kwargs.get("use_standardization", False):
        config["preprocessor"] = {
            "type": "standardization",
            "method": kwargs.get("standardization_method", "standard")
        }
    
    # リターン算出器の設定
    if kwargs.get("return_type"):
        config["return_calculator"] = {
            "type": kwargs["return_type"]
        }
    
    return config


# 実用例
def create_examples():
    """実用例の作成"""
    
    # 通貨ファクター研究の設定例
    currency_factor_config = create_research_pipeline_config(
        "factor_research",
        factor_name="JPY_sensitivity",
        factor_type="currency_sensitivity",
        factor_definition={
            "JPY_positive": ["2413", "3141", "4587"],
            "JPY_negative": ["7283", "7296", "5988"]
        },
        return_type="intraday",
        pca_components=1
    )
    
    # マルチ戦略分析の設定例
    multi_strategy_config = create_multi_strategy_config(
        strategies=["momentum", "volatility", "technical", "ranking"],
        momentum_windows=[5, 10, 21],
        volatility_windows=[5, 10, 21],
        technical_indicators=["rsi", "macd", "bb"],
        use_pca=True,
        pca_components=2
    )
    
    return {
        "currency_factor": currency_factor_config,
        "multi_strategy": multi_strategy_config
    }


if __name__ == "__main__":
    # 使用例
    print("パイプラインファクトリーの使用例")
    
    # プリセットパイプラインの作成
    basic_pipeline = create_preset_pipeline("basic_features")
    print(f"基本パイプライン: {basic_pipeline}")
    
    # 設定ベースのパイプライン作成
    research_config = create_research_pipeline_config(
        "sector_rotation",
        momentum_windows=[5, 21],
        ranking_columns=["5d_mom", "21d_mom"]
    )
    
    research_pipeline = create_pipeline_from_config(research_config)
    print(f"研究パイプライン: {research_pipeline}")
    
    # パイプライン管理者の使用
    manager = get_pipeline_manager()
    available_pipelines = manager.list_available_pipelines()
    print(f"利用可能なパイプライン: {available_pipelines}")