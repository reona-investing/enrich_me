import yaml

def yaml_loader(yaml_path: str) -> list[dict[str, str]]:
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

if __name__ == '__main__':
    from utils.paths import Paths
    yl = yaml_loader(Paths.RAW_STOCK_PRICE_COLUMNS_YAML)
    print(yl)
    print(type(yl))