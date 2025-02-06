import pandas as pd


def dtypes_converter(yaml_info: dict[str, dict[str|str]] | list[dict[str, str]], df: pd.DataFrame,
                     name_key: str = 'fixed_name', dtype_key: str = 'fixed_dtype',
                     yaml_listname: str = 'columns') -> pd.DataFrame:
    """
    設定ファイルをもとに各カラムに適切なデータ型を設定。

    Args:
        yaml_info (dict[str, dict[str|str]] | list[dict[str, str]]): yamlから読み込んだ設定情報
        df (pd.DataFrame): カラム名変換後のデータ
        name_col (str): yaml内でカラム名を規定したキー
        dtype_col (str): yaml内でデータ型を規定したキー
        yaml_listname (str): `yaml_info` が辞書の場合、検索対象のリスト名

    Returns:
        pd.DataFrame: データ型変換後の財務データ
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

    convert_dict_except_datetime = {col[name_key]: eval(col[dtype_key]) \
                                    for col in yaml_data if col[dtype_key] != 'datetime'} 
    datetime_columns = [col[name_key] \
                        for col in yaml_data if col[dtype_key] == 'datetime']

    df[[x for x in convert_dict_except_datetime.keys()]] = \
        df[[x for x in convert_dict_except_datetime.keys()]].astype(convert_dict_except_datetime)

    for column in datetime_columns:
        df[column]= df[column].astype(str).str[:10]
        df[column] = pd.to_datetime(df[column])
    
    return df