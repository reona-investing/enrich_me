import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from preprocessing.methods.base_preprocessor import BasePreprocessor
from utils.timeseries import Duration


class YourPreprocessorName(BasePreprocessor):
    """
    [概要]
    この前処理クラスは ○○ を行います。

    Parameters
    ----------
    param1 : type
        パラメータの説明
    ...
    """

    def __init__(self,
                 *,
                 copy: bool = True,
                 fit_duration: Optional[Duration] = None,
                 time_column: str):
        super().__init__(copy=copy, fit_duration=fit_duration, time_column=time_column)
        
        # 独自パラメータをここに定義
        # self.example_param = example_param

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'YourPreprocessorName':
        """
        fit処理：指定期間のデータでパラメータを学習
        [必須呼び出し]
            - self._validate_input(X)
            - self.fit_duration.extract_from_df(X, self.time_column)
            - self._store_fit_metadata(X)
            - self._mark_as_fitted(...)  # 必ず最後に
        """
        self._validate_input(X)
        if self.fit_duration is not None:
            X_fit = self.fit_duration.extract_from_df(X, self.time_column)
            if X_fit.empty:
                raise ValueError(
                    f"指定された期間({self.fit_duration.start}～{self.fit_duration.end})のデータが存在しません。"
                )
        else:
            X_fit = X
        self._store_fit_metadata(X)

        # ここにfit用処理を書く

        self._mark_as_fitted(
            # 例: scaler_=self.scaler_
        )
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        transform処理：全期間データに対して前処理を適用
        [必須呼び出し]
            - self._check_is_fitted()
            - self._validate_input(X)
            - self._prepare_output(X)
        """
        self._check_is_fitted()
        self._validate_input(X)
        X_transformed = self._prepare_output(X)

        # ここに変換処理を書く

        return X_transformed
