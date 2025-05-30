from sklearn.pipeline import Pipeline
import pandas as pd
from typing import Any, List, Optional

class PreprocessingPipeline:
    def __init__(self, steps: List[tuple[str, Any]]):
        """
        sklearn.pipeline.Pipeline をラップするクラス
        
        Parameters
        ----------
        steps : List[tuple[str, Transformer]]
            sklearnと同様の形式で処理ステップを渡す
        """
        self.pipeline = Pipeline(steps)
        self.input_columns_: Optional[List[str]] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'PreprocessingPipeline':
        self._validate_input(X)
        self.input_columns_ = X.columns.tolist()
        self.pipeline.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fitted()
        X_trans = self.pipeline.transform(X)
        # DataFrameに戻す（必要に応じて名前を推定/復元）
        if isinstance(X_trans, pd.DataFrame):
            return X_trans
        return pd.DataFrame(X_trans, index=X.index)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _validate_input(self, X: Any) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

    def _check_is_fitted(self) -> None:
        if self.input_columns_ is None:
            raise ValueError("PipelineWrapper is not fitted yet")

    def get_pipeline(self) -> Pipeline:
        """内部のPipelineインスタンスを取得"""
        return self.pipeline

    def get_feature_names_out(self) -> List[str]:
        """最終ステップの出力カラム名を取得できる場合は取得"""
        last_step = self.pipeline.steps[-1][1]
        if hasattr(last_step, "get_feature_names_out"):
            return last_step.get_feature_names_out()
        return [f"f{i}" for i in range(self.pipeline.steps[-1][1].n_components)]  # 例：PCAなど