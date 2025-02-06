import yaml

def yaml_loader(yaml_path: str, key: str) -> dict:
    '''
    yamlファイルを読み込みます。
    Args:
        yaml_path (str): 読み込み対象のyamlファイルのパス
        key (str): YAMLの大元のキー
    Returns:
        dict: yamlファイル内の情報
    '''
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        yaml_info = list(yaml.safe_load_all(yaml_file))
    return next((d for d in yaml_info if key in list(d.keys())[0]), None)


def including_columns_loader(yaml_path: str, key: str, including_key: str = 'include') -> list[dict[str, str]]:
    """
    財務データのカラム情報を YAML からロードし、使用するカラムのみを抽出する。

    Args:
        yaml_path (str): YAML ファイルのパス
        key (str): YAMLの大元のキー
        including_key (str): YAMLの各要素を採用するかどうかのキー

    Returns:
        dict[str: list[dict[str, str]]]: 設定されたカラム情報のリスト
    """
    columns_info = yaml_loader(yaml_path, key)
    return [col for col in columns_info[key] if col[including_key]]


def including_columns_loader(yaml_path: str, key: str, including_key: str = 'include') -> list[dict[str, str]]:
    """
    財務データのカラム情報を YAML からロードし、使用するカラムのみを抽出する。

    Args:
        yaml_path (str): YAML ファイルのパス
        key (str): YAMLの大元のキー
        including_key (str): YAMLの各要素を採用するかどうかのキー

    Returns:
        dict[str: list[dict[str, str]]]: 設定されたカラム情報のリスト
    """
    columns_info = yaml_loader(yaml_path, key)
    return [col for col in columns_info[key] if col[including_key]]