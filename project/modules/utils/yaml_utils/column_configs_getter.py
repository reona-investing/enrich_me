from typing import Any, Dict, List
from utils.yaml_utils.yaml_loader import yaml_loader

class ColumnConfigsGetter:
    def __init__(self, cols_yaml_path: str):
        """
        初期化処理として、YAML ファイルを読み込む。

        Args:
            cols_yaml_path (str): YAML ファイルのパス
        """
        self.column_config_info = yaml_loader(cols_yaml_path)

    def get_any_column_info(self, key: str, info_name: str) -> str | None:
        """
        YAML ファイルから読み込んだリストを検索し、特定のキーに対応する値を取得する。

        Args:
            key (str): 検索対象とするキーの値
            info_name (str): 取得する要素名

        Returns:
            str | None: 検索条件に一致する要素の `target_key` の値 (見つからなかった場合は `None`)
        """
        return next((item[info_name] for item in self.column_config_info if item.get('key') == key), None)

    def get_all_columns_info_asdict(self, info_name: str) -> dict:
        return {item['key']: item[info_name] for item in self.column_config_info}
    
    def get_column_name(self, key: str) -> str | None:
        return self.get_any_column_info(key, 'name')

    def get_column_names(self, keys: list[str]) -> list[str | None]:
        return [self.get_any_column_info(key, 'name') for key in keys]

    def get_all_columns_name_asdict(self) -> dict[str, str]:
        return self.get_all_columns_info_asdict('name')    
    
    def get_column_dtype(self, key: str) -> str | None:
        return self.get_any_column_info(key, 'dtype')

    def get_column_dtypes(self, keys: list[str]) -> list[str | None]:
        return [self.get_any_column_info(key, 'dtype') for key in keys]

def column_name_getter(yaml_info: Dict[str, List[Dict[str, Any]]] | List[Dict[str, Any]], 
                    search_condition: Dict[str, Any], target_key: str, yaml_listname: str = 'columns') -> str | None:
    """
    YAML ファイルから読み込んだ辞書またはリストを検索し、特定のキーに対応する値を取得する。

    Args:
        yaml_info (dict[str, list[dict]] | list[dict]): YAML ファイルから読み込んだデータ
            - 辞書 (dict[str, list[dict]]): 指定のリスト (`yaml_listname`) を含む辞書
            - リスト (list[dict]): 辞書のリスト
        search_condition (dict[str, Any]): 検索条件となるキーと値の辞書
            - 例: `{"raw_name": "LocalCode"}`
        target_key (str): 取得したい値が格納されているキー名
        yaml_listname (str, optional): `yaml_info` が辞書の場合、検索対象のリスト名

    Returns:
        str | None: 検索条件に一致する要素の `target_key` の値 (見つからなかった場合は `None`)

    Raises:
        ValueError: `yaml_info` が辞書の場合、`yaml_listname` が指定されていない場合
        TypeError: `yaml_info` の型が `dict` または `list` でない場合
    """
    if isinstance(yaml_info, dict):
        if yaml_listname is None:
            raise ValueError("yaml_info に辞書を指定する場合、yaml_listname は入力必須です。")
        if yaml_listname not in yaml_info or not isinstance(yaml_info[yaml_listname], list):
            raise ValueError(f"指定された yaml_listname '{yaml_listname}' が辞書内に存在しないか、リスト形式ではありません。")
        yaml_data = yaml_info[yaml_listname]
    elif isinstance(yaml_info, list):
        yaml_data = yaml_info
    else:
        raise TypeError("yaml_info には YAML ファイルから読み込んだ辞書 (dict[str, list[dict]]) またはリスト (list[dict]) を指定してください。")

    search_key = list(search_condition.keys())[0]
    search_value = list(search_condition.values())[0]

    return next((item[target_key] for item in yaml_data if item.get(search_key) == search_value), None)


if __name__ =='__main__':
    from utils.paths import Paths
    ccg = ColumnConfigsGetter(Paths.RAW_STOCK_PRICE_COLUMNS_YAML)
    print(ccg.get_column_name('日付'))