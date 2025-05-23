"""
パイプライン統合システム
"""
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
import warnings

from ..return_calculators.base import ReturnCalculator
from ..preprocessors.base import Preprocessor
from ..transformers.base import Transformer
from ..contracts.input_contracts import BaseInputContract
from ..contracts.output_contracts import BaseOutputContract


class CalculationPipeline:
    """計算パイプラインの統合クラス"""
    
    def __init__(self, 
                 input_contract: Optional[BaseInputContract] = None,
                 output_contract: Optional[BaseOutputContract] = None):
        self.input_contract = input_contract
        self.output_contract = output_contract
        self.return_calculator: Optional[ReturnCalculator] = None
        self.preprocessor: Optional[Preprocessor] = None
        self.transformers: List[Transformer] = []
        self._execution_metadata = {}
        self._intermediate_results = {}
    
    def set_return_calculator(self, calculator: ReturnCalculator) -> 'CalculationPipeline':
        """リターン算出器を設定"""
        self.return_calculator = calculator
        return self
    
    def set_preprocessor(self, preprocessor: Preprocessor) -> 'CalculationPipeline':
        """前処理器を設定"""
        self.preprocessor = preprocessor
        return self
    
    def add_transformer(self, transformer: Transformer) -> 'CalculationPipeline':
        """変換器を追加"""
        self.transformers.append(transformer)
        return self
    
    def remove_transformer(self, index: int) -> 'CalculationPipeline':
        """変換器を削除"""
        if 0 <= index < len(self.transformers):
            self.transformers.pop(index)
        return self
    
    def clear_transformers(self) -> 'CalculationPipeline':
        """全ての変換器をクリア"""
        self.transformers.clear()
        return self
    
    def execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        パイプラインを実行
        
        Args:
            df: 入力データフレーム
            **kwargs: 各段階のパラメータ
            
        Returns:
            処理済みデータフレーム
        """
        start_time = datetime.now()
        self._intermediate_results.clear()
        
        try:
            # 1. 入力検証
            current_df = self._validate_input(df)
            self._intermediate_results['input'] = current_df.copy()
            
            # 2. リターン算出
            if self.return_calculator:
                calc_params = kwargs.get('return_calculator_params', {})
                current_df = self.return_calculator.execute(current_df, **calc_params)
                self._intermediate_results['return_calculation'] = current_df.copy()
            
            # 3. 前処理適用
            if self.preprocessor:
                prep_params = kwargs.get('preprocessor_params', {})
                current_df = self.preprocessor.execute(current_df, prep_params)
                self._intermediate_results['preprocessing'] = current_df.copy()
            
            # 4. 変換処理適用
            for i, transformer in enumerate(self.transformers):
                trans_params = kwargs.get(f'transformer_{i}_params', {})
                transformer_result = transformer.execute(current_df, **trans_params)
                
                # 新しいカラムのみを追加（既存カラムは保持）
                new_columns = set(transformer_result.columns) - set(current_df.columns)
                if new_columns:
                    current_df = pd.concat([current_df, transformer_result[list(new_columns)]], axis=1)
                
                self._intermediate_results[f'transformer_{i}'] = current_df.copy()
            
            # 5. 出力検証・整形
            result = self._validate_output(current_df)
            
            # 実行メタデータを記録
            end_time = datetime.now()
            self._execution_metadata = {
                'start_time': start_time,
                'end_time': end_time,
                'execution_time': (end_time - start_time).total_seconds(),
                'input_shape': df.shape,
                'output_shape': result.shape,
                'stages_executed': self._get_executed_stages(),
                'parameters': kwargs
            }
            
            return result
            
        except Exception as e:
            # エラー時のメタデータ記録
            self._execution_metadata = {
                'start_time': start_time,
                'end_time': datetime.now(),
                'error': str(e),
                'error_type': type(e).__name__,
                'stages_executed': self._get_executed_stages(),
                'parameters': kwargs
            }
            raise
    
    def _validate_input(self, df: pd.DataFrame) -> pd.DataFrame:
        """入力データの検証"""
        if self.input_contract:
            self.input_contract.validate(df)
        
        # 基本的な検証
        if df.empty:
            raise ValueError("入力データが空です")
        
        return df
    
    def _validate_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """出力データの検証と整形"""
        if self.output_contract:
            self.output_contract.validate_output(df)
            return self.output_contract.format_output(df)
        
        # デフォルトの整形
        return df.sort_index()
    
    def _get_executed_stages(self) -> List[str]:
        """実行された段階のリストを取得"""
        stages = ['input_validation']
        
        if self.return_calculator:
            stages.append('return_calculation')
        if self.preprocessor:
            stages.append('preprocessing')
        
        for i in range(len(self.transformers)):
            stages.append(f'transformer_{i}')
        
        stages.append('output_validation')
        return stages
    
    def get_execution_metadata(self) -> Dict[str, Any]:
        """実行メタデータを取得"""
        return self._execution_metadata.copy()
    
    def get_intermediate_result(self, stage: str) -> pd.DataFrame:
        """
        中間結果を取得
        
        Args:
            stage: 段階名 ('input', 'return_calculation', 'preprocessing', 'transformer_0', etc.)
            
        Returns:
            指定段階での結果
        """
        if stage not in self._intermediate_results:
            available_stages = list(self._intermediate_results.keys())
            raise ValueError(f"段階 '{stage}' は存在しません。利用可能: {available_stages}")
        
        return self._intermediate_results[stage].copy()
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        パイプライン構成の検証
        
        Returns:
            検証結果
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'configuration': self.get_configuration_summary()
        }
        
        # リターン算出器の検証
        if not self.return_calculator:
            validation_results['warnings'].append("リターン算出器が設定されていません")
        
        # 前処理器の検証
        if self.preprocessor is None:
            validation_results['warnings'].append("前処理器が設定されていません（NoPreprocessorが使用されます）")
        
        # 変換器の検証
        if not self.transformers:
            validation_results['warnings'].append("変換器が設定されていません")
        
        # パイプライン内の整合性チェック
        if self.return_calculator and self.preprocessor:
            # リターン算出器の出力と前処理器の入力の整合性
            calc_output_cols = getattr(self.return_calculator, 'output_column_name', None)
            prep_input_cols = getattr(self.preprocessor, 'required_input_columns', [])
            
            if calc_output_cols and prep_input_cols:
                if calc_output_cols not in prep_input_cols:
                    validation_results['warnings'].append(
                        f"リターン算出器の出力 '{calc_output_cols}' が前処理器の必要入力に含まれていません"
                    )
        
        # エラーがあれば無効とマーク
        if validation_results['errors']:
            validation_results['is_valid'] = False
        
        return validation_results
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """パイプライン構成の要約を取得"""
        return {
            'return_calculator': {
                'class': self.return_calculator.__class__.__name__ if self.return_calculator else None,
                'output_column': getattr(self.return_calculator, 'output_column_name', None)
            },
            'preprocessor': {
                'class': self.preprocessor.__class__.__name__ if self.preprocessor else None,
                'is_fitted': getattr(self.preprocessor, 'is_fitted', False)
            },
            'transformers': [
                {
                    'index': i,
                    'class': transformer.__class__.__name__,
                    'output_columns': getattr(transformer, 'output_column_names', [])
                }
                for i, transformer in enumerate(self.transformers)
            ],
            'total_stages': len(self._get_executed_stages())
        }
    
    def clone(self) -> 'CalculationPipeline':
        """パイプラインのコピーを作成"""
        new_pipeline = CalculationPipeline(
            input_contract=self.input_contract,
            output_contract=self.output_contract
        )
        
        if self.return_calculator:
            new_pipeline.return_calculator = self.return_calculator
        
        if self.preprocessor:
            new_pipeline.preprocessor = self.preprocessor
        
        new_pipeline.transformers = self.transformers.copy()
        
        return new_pipeline
    
    def save_configuration(self, filepath: str) -> None:
        """パイプライン構成を保存"""
        import json
        
        config = {
            'return_calculator': {
                'class': self.return_calculator.__class__.__name__ if self.return_calculator else None,
                'module': self.return_calculator.__class__.__module__ if self.return_calculator else None
            },
            'preprocessor': {
                'class': self.preprocessor.__class__.__name__ if self.preprocessor else None,
                'module': self.preprocessor.__class__.__module__ if self.preprocessor else None,
                'is_fitted': getattr(self.preprocessor, 'is_fitted', False)
            },
            'transformers': [
                {
                    'class': transformer.__class__.__name__,
                    'module': transformer.__class__.__module__
                }
                for transformer in self.transformers
            ],
            'contracts': {
                'input_contract': self.input_contract.__class__.__name__ if self.input_contract else None,
                'output_contract': self.output_contract.__class__.__name__ if self.output_contract else None
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def __repr__(self) -> str:
        """パイプラインの文字列表現"""
        stages = []
        
        if self.return_calculator:
            stages.append(f"ReturnCalc({self.return_calculator.__class__.__name__})")
        
        if self.preprocessor:
            stages.append(f"Preprocess({self.preprocessor.__class__.__name__})")
        
        for i, transformer in enumerate(self.transformers):
            stages.append(f"Transform{i}({transformer.__class__.__name__})")
        
        return f"CalculationPipeline({' -> '.join(stages)})"


class BatchPipeline:
    """バッチ処理用パイプライン"""
    
    def __init__(self, base_pipeline: CalculationPipeline):
        self.base_pipeline = base_pipeline
        self._batch_results = {}
        self._batch_metadata = {}
    
    def execute_batch(self, data_dict: Dict[str, pd.DataFrame], 
                     config_dict: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, pd.DataFrame]:
        """
        複数データに対してバッチ実行
        
        Args:
            data_dict: データ名をキーとするデータフレーム辞書
            config_dict: データ別の設定辞書
            
        Returns:
            実行結果の辞書
        """
        start_time = datetime.now()
        results = {}
        errors = {}
        
        for data_name, data_df in data_dict.items():
            try:
                # データ別の設定を取得
                data_config = config_dict.get(data_name, {}) if config_dict else {}
                
                # パイプライン実行
                result = self.base_pipeline.execute(data_df, **data_config)
                results[data_name] = result
                
            except Exception as e:
                errors[data_name] = {
                    'error': str(e),
                    'error_type': type(e).__name__
                }
                warnings.warn(f"データ '{data_name}' の処理に失敗しました: {e}")
        
        # バッチメタデータを記録
        end_time = datetime.now()
        self._batch_metadata = {
            'start_time': start_time,
            'end_time': end_time,
            'total_execution_time': (end_time - start_time).total_seconds(),
            'total_datasets': len(data_dict),
            'successful_datasets': len(results),
            'failed_datasets': len(errors),
            'errors': errors
        }
        
        self._batch_results = results
        return results
    
    def get_batch_metadata(self) -> Dict[str, Any]:
        """バッチ実行のメタデータを取得"""
        return self._batch_metadata.copy()
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """バッチ実行の要約を取得"""
        if not self._batch_results:
            return {
                "error": "バッチが実行されていないか、結果が空です",
                "execution_metadata": self._batch_metadata
            }
        
        summary = {
            'total_datasets': len(self._batch_results),
            'dataset_shapes': {name: df.shape for name, df in self._batch_results.items()},
            'total_rows': sum(df.shape[0] for df in self._batch_results.values()),
            'total_columns': sum(df.shape[1] for df in self._batch_results.values()),
            'execution_metadata': self._batch_metadata
        }
        
        return summary