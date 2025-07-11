# preprocessing/methods/template のクラス仕様書

## template.py

### class YourPreprocessorName
[概要]
この前処理クラスは ○○ を行います。

Parameters
----------
param1 : type
    パラメータの説明
...
 - __init__:
 - fit: fit処理：指定期間のデータでパラメータを学習
[必須呼び出し]
    - self._validate_input(X)
    - self.fit_duration.extract_from_df(X, self.time_column)
    - self._store_fit_metadata(X)
    - self._mark_as_fitted(...)  # 必ず最後に
- transform: transform処理：全期間データに対して前処理を適用
[必須呼び出し]
    - self._check_is_fitted()
    - self._validate_input(X)
    - self._prepare_output(X)

