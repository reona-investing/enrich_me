# utils/yaml_utils のクラス仕様書

## column_configs_getter.py

### class ColumnConfigsGetter
- __init__: 初期化処理として、YAML ファイルを読み込む。

Args:
    cols_yaml_path (str): YAML ファイルのパス
- get_any_column_info: YAML ファイルから読み込んだリストを検索し、特定のキーに対応する値を取得する。

Args:
    key (str): 検索対象とするキーの値
    info_name (str): 取得する要素名

Returns:
    str | None: 検索条件に一致する要素の `target_key` の値 (見つからなかった場合は `None`)
- get_all_columns_info_asdict: 
- get_column_name: 
- get_column_names: 
- get_all_columns_name_asdict: 
- get_column_dtype: 
- get_column_dtypes: 

