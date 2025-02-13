import yaml

def yaml_loader(yaml_path: str) -> list:
    '''
    yamlファイルを読み込み、中身をリストとして返します。
    Args:
        yaml_path (str): 読み込み対象のyamlファイルのパス
    Returns:
        list: yamlファイル内の情報
    '''
    with open(yaml_path, 'r', encoding='utf-8') as yaml_file:
        yaml_info = yaml.safe_load(yaml_file)
    return list(yaml_info.values())[0]

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

if __name__ == '__main__':
    from utils.paths import Paths
    yl = yaml_loader(Paths.RAW_STOCK_PRICE_COLUMNS_YAML)
    print(yl)
    print(type(yl))