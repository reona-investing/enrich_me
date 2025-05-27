"""
パイプライン設定管理システム
"""
import yaml
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime


class PipelineConfigManager:
    """パイプライン設定の管理クラス"""
    
    def __init__(self, config_directory: str = "config_and_settings/pipeline_configs"):
        self.config_directory = Path(config_directory)
        self.config_directory.mkdir(parents=True, exist_ok=True)
        self._default_configs = self._create_default_configs()
    
    def save_config(self, config_name: str, config: Dict[str, Any], 
                   config_type: str = "yaml") -> None:
        """設定を保存"""
        filename = f"{config_name}.{config_type}"
        filepath = self.config_directory / filename
        
        # メタデータを追加
        config_with_meta = {
            "metadata": {
                "name": config_name,
                "created_at": datetime.now().isoformat(),
                "config_type": config_type,
                "version": "1.0"
            },
            "config": config
        }
        
        if config_type.lower() == "yaml":
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config_with_meta, f, default_flow_style=False, 
                         allow_unicode=True, indent=2)
        elif config_type.lower() == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_with_meta, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"未対応の設定ファイル形式: {config_type}")
    
    def load_config(self, config_name: str, config_type: str = "yaml") -> Dict[str, Any]:
        """設定を読み込み"""
        filename = f"{config_name}.{config_type}"
        filepath = self.config_directory / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {filepath}")
        
        if config_type.lower() == "yaml":
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
        elif config_type.lower() == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
        else:
            raise ValueError(f"未対応の設定ファイル形式: {config_type}")
        
        # メタデータと設定を分離
        if isinstance(loaded, dict) and "config" in loaded:
            return loaded["config"]
        else:
            return loaded
    
    def list_configs(self) -> List[Dict[str, str]]:
        """利用可能な設定ファイル一覧を取得"""
        configs = []
        
        for filepath in self.config_directory.glob("*.yaml"):
            configs.append({
                "name": filepath.stem,
                "type": "yaml",
                "path": str(filepath)
            })
        
        for filepath in self.config_directory.glob("*.json"):
            configs.append({
                "name": filepath.stem,
                "type": "json",
                "path": str(filepath)
            })
        
        return configs
    
    def get_default_config(self, config_type: str) -> Dict[str, Any]:
        """デフォルト設定を取得"""
        if config_type not in self._default_configs:
            raise ValueError(f"未対応の設定タイプ: {config_type}")
        return self._default_configs[config_type].copy()
    
    def _create_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """デフォルト設定を作成"""
        return {
            "basic_features": {
                "return_calculator": {
                    "type": "daily",
                    "handle_missing": "ffill",
                    "detect_outliers": False
                },
                "preprocessor": {
                    "type": "none"
                },
                "transformers": [
                    {
                        "type": "momentum",
                        "windows": [5, 21],
                        "base_column": "1d_return",
                        "add_rank": True
                    },
                    {
                        "type": "volatility",
                        "windows": [5, 21], 
                        "base_column": "1d_return",
                        "add_rank": True
                    }
                ]
            },
            
            "pca_features": {
                "return_calculator": {
                    "type": "intraday"
                },
                "preprocessor": {
                    "type": "pca_market_factor_removal",
                    "n_components": 1,
                    "train_start": "2020-01-01",
                    "train_end": "2022-12-31"
                },
                "transformers": [
                    {
                        "type": "momentum",
                        "windows": [5, 21],
                        "base_column": "Target",
                        "add_rank": True
                    }
                ]
            },
            
            "technical_analysis": {
                "return_calculator": {
                    "type": "daily"
                },
                "preprocessor": {
                    "type": "standardization",
                    "method": "standard"
                },
                "transformers": [
                    {
                        "type": "technical_indicators",
                        "indicators": ["rsi", "macd", "bb"],
                        "price_columns": ["Close"]
                    },
                    {
                        "type": "momentum",
                        "windows": [5, 10, 21],
                        "add_rank": True
                    },
                    {
                        "type": "volatility",
                        "windows": [5, 10, 21],
                        "add_rank": True
                    }
                ]
            },
            
            "currency_factor": {
                "data_generator": {
                    "type": "custom_factor",
                    "factor_definition": {
                        "JPY_positive": ["2413", "3141", "4587", "1835", "4684"],
                        "JPY_negative": ["7283", "7296", "5988", "8015", "7278"]
                    }
                },
                "return_calculator": {
                    "type": "intraday"
                },
                "preprocessor": {
                    "type": "pca_market_factor_removal",
                    "n_components": 1
                },
                "transformers": [
                    {
                        "type": "momentum",
                        "windows": [5, 21],
                        "add_rank": True
                    },
                    {
                        "type": "volatility",
                        "windows": [5, 21],
                        "add_rank": True
                    }
                ]
            },
            
            "multi_timeframe": {
                "return_calculator": {
                    "type": "multi_period",
                    "periods": [1, 5, 21, 60],
                    "return_type": "daily"
                },
                "preprocessor": {
                    "type": "outlier_removal",
                    "method": "iqr",
                    "threshold": 1.5
                },
                "transformers": [
                    {
                        "type": "ranking",
                        "target_columns": ["1d_return", "5d_return", "21d_return", "60d_return"]
                    }
                ]
            }
        }


class ConfigTemplateGenerator:
    """設定テンプレート生成器"""
    
    @staticmethod
    def generate_basic_template() -> Dict[str, Any]:
        """基本的な設定テンプレート"""
        return {
            "pipeline_name": "custom_pipeline",
            "description": "カスタムパイプライン設定",
            "return_calculator": {
                "type": "daily",  # daily, intraday, overnight, bond, commodity_jpy
                "handle_missing": "ffill",  # drop, ffill, bfill, none
                "detect_outliers": False
            },
            "preprocessor": {
                "type": "none",  # none, pca_market_factor_removal, standardization, outlier_removal
                # PCA用パラメータ
                "n_components": 1,
                "train_start": "2020-01-01",
                "train_end": "2022-12-31",
                # 標準化用パラメータ
                "method": "standard"  # standard, minmax, robust
            },
            "transformers": [
                {
                    "type": "momentum",
                    "windows": [5, 21],
                    "base_column": "1d_return",
                    "add_rank": True,
                    "exclude_current_day": True
                },
                {
                    "type": "volatility",
                    "windows": [5, 21],
                    "base_column": "1d_return", 
                    "add_rank": True,
                    "exclude_current_day": True
                }
            ]
        }
    
    @staticmethod
    def generate_factor_research_template() -> Dict[str, Any]:
        """ファクター研究用テンプレート"""
        return {
            "pipeline_name": "factor_research",
            "description": "ファクター研究用パイプライン",
            "data_generator": {
                "type": "custom_factor",
                "factor_definition": {
                    "factor_positive": ["CODE1", "CODE2", "CODE3"],
                    "factor_negative": ["CODE4", "CODE5", "CODE6"]
                }
            },
            "return_calculator": {
                "type": "intraday"
            },
            "preprocessor": {
                "type": "pca_market_factor_removal",
                "n_components": 1,
                "train_start": "2020-01-01",
                "train_end": "2022-12-31"
            },
            "transformers": [
                {
                    "type": "momentum",
                    "windows": [5, 10, 21],
                    "base_column": "Target",
                    "add_rank": True
                },
                {
                    "type": "volatility",
                    "windows": [5, 10, 21],
                    "base_column": "Target",
                    "add_rank": True
                },
                {
                    "type": "technical_indicators",
                    "indicators": ["rsi", "macd"],
                    "price_columns": ["Close"]
                }
            ]
        }
    
    @staticmethod
    def generate_currency_research_template() -> Dict[str, Any]:
        """通貨研究用テンプレート"""
        return {
            "pipeline_name": "currency_research",
            "description": "通貨感応度分析用パイプライン",
            "return_calculator": {
                "type": "daily"
            },
            "transformers": [
                {
                    "type": "currency_relative_strength",
                    "return_suffix": "_1d_return"
                },
                {
                    "type": "momentum",
                    "windows": [5, 21],
                    "base_column": "JPY_1d_return",
                    "add_rank": True
                }
            ]
        }


class ConfigValidator:
    """設定の検証クラス"""
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Dict[str, List[str]]:
        """設定の妥当性を検証"""
        errors = []
        warnings = []
        
        # 必須フィールドの確認
        if "return_calculator" not in config and "transformers" not in config:
            errors.append("return_calculatorまたはtransformersのいずれかが必要です")
        
        # リターン算出器の検証
        if "return_calculator" in config:
            calc_config = config["return_calculator"]
            valid_types = ["daily", "intraday", "overnight", "bond", "commodity_jpy", "multi_period"]
            if calc_config.get("type") not in valid_types:
                errors.append(f"無効なリターン算出器タイプ: {calc_config.get('type')}")
        
        # 前処理器の検証
        if "preprocessor" in config:
            prep_config = config["preprocessor"]
            valid_types = ["none", "pca_market_factor_removal", "standardization", "outlier_removal"]
            if prep_config.get("type") not in valid_types:
                errors.append(f"無効な前処理器タイプ: {prep_config.get('type')}")
            
            # PCA特有の検証
            if prep_config.get("type") == "pca_market_factor_removal":
                if "train_start" not in prep_config or "train_end" not in prep_config:
                    errors.append("PCAには訓練期間の指定が必要です")
        
        # 変換器の検証
        if "transformers" in config:
            for i, trans_config in enumerate(config["transformers"]):
                valid_types = ["momentum", "volatility", "ranking", "technical_indicators", 
                              "currency_relative_strength", "bond_spread", "sector_categorical"]
                if trans_config.get("type") not in valid_types:
                    errors.append(f"変換器{i}: 無効なタイプ {trans_config.get('type')}")
                
                # モメンタム・ボラティリティの検証
                if trans_config.get("type") in ["momentum", "volatility"]:
                    if "windows" not in trans_config:
                        warnings.append(f"変換器{i}: windowsが指定されていません")
        
        return {"errors": errors, "warnings": warnings}


# 便利関数
def load_pipeline_config(config_name: str, config_directory: Optional[str] = None) -> Dict[str, Any]:
    """パイプライン設定を読み込み"""
    manager = PipelineConfigManager(config_directory) if config_directory else PipelineConfigManager()
    return manager.load_config(config_name)


def save_pipeline_config(config_name: str, config: Dict[str, Any], 
                        config_directory: Optional[str] = None) -> None:
    """パイプライン設定を保存"""
    manager = PipelineConfigManager(config_directory) if config_directory else PipelineConfigManager()
    manager.save_config(config_name, config)


def create_config_from_template(template_type: str = "basic") -> Dict[str, Any]:
    """テンプレートから設定を作成"""
    if template_type == "basic":
        return ConfigTemplateGenerator.generate_basic_template()
    elif template_type == "factor_research":
        return ConfigTemplateGenerator.generate_factor_research_template()
    elif template_type == "currency_research":
        return ConfigTemplateGenerator.generate_currency_research_template()
    else:
        raise ValueError(f"未対応のテンプレートタイプ: {template_type}")


def validate_pipeline_config(config: Dict[str, Any]) -> bool:
    """パイプライン設定を検証"""
    validation_result = ConfigValidator.validate_config(config)
    
    if validation_result["errors"]:
        print("設定エラー:")
        for error in validation_result["errors"]:
            print(f"  - {error}")
        return False
    
    if validation_result["warnings"]:
        print("設定警告:")
        for warning in validation_result["warnings"]:
            print(f"  - {warning}")
    
    return True


# デフォルト設定の保存
def initialize_default_configs():
    """デフォルト設定ファイルを初期化"""
    manager = PipelineConfigManager()
    
    # デフォルト設定を保存
    for config_name, config in manager._default_configs.items():
        try:
            manager.save_config(config_name, config)
            print(f"デフォルト設定を保存しました: {config_name}")
        except Exception as e:
            print(f"設定保存エラー {config_name}: {e}")


if __name__ == "__main__":
    # デフォルト設定の初期化
    initialize_default_configs()
    
    # 設定例の表示
    print("\n利用可能なデフォルト設定:")
    manager = PipelineConfigManager()
    for config_name in manager._default_configs.keys():
        print(f"  - {config_name}")
    
    # テンプレート例の作成
    print("\n基本テンプレートの例:")
    basic_template = create_config_from_template("basic")
    print(f"  パイプライン名: {basic_template['pipeline_name']}")
    print(f"  変換器数: {len(basic_template['transformers'])}")