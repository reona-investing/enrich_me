import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from preprocessing.methods.base_preprocessor import BasePreprocessor


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
                 copy: bool = True,
                 fit_start: Union[str, pd.Timestamp, None] = None,
                 fit_end: Union[str, pd.Timestamp, None] = None,
                 time_column: Optional[str] = 'Date'):
        super().__init__(copy=copy, fit_start=fit_start, fit_end=fit_end, time_column=time_column)
        
        # 独自パラメータをここに定義
        # self.example_param = example_param

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[Any] = None) -> 'YourPreprocessorName':
        """
        fit処理：指定期間のデータでパラメータを学習
        [必須呼び出し]
            - self._validate_input(X)
            - self._filter_data_by_time(X, self.fit_start, self.fit_end)
            - self._store_fit_metadata(X)
            - self._mark_as_fitted(...)  # 必ず最後に
        """
        self._validate_input(X)
        X_fit = self._filter_data_by_time(X, self.fit_start, self.fit_end)
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
