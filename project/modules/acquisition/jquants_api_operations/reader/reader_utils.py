import pandas as pd
  

def filter_stocks(df:pd.DataFrame, list_df: pd.DataFrame, 
                  filter:str | None, filtered_code_list:list[str] | None, code_col: str | None) -> pd.DataFrame: # 対象銘柄の抜き取り
    """
    対象銘柄を抜き取ります。filterとfiltered_code_listがどちらも設定されている場合、filterの条件が優先されます。
    Args:
        df (pd.DataFrame): 抜き取り対象のデータフレーム
        list_df (pd.DataFrame): 銘柄リストのデータフレーム（抜き取り用）
        filter (str): 銘柄の絞り込み条件をstr型で指定
        filtered_code_list (list[str]): 絞り込み対象の銘柄をリスト型で指定
        code_col (str): データフレーム内の銘柄コード列の列名
    Returns:
        pd.DataFrame: 絞り込み後のデータフレーム
    """
    if filter:
        filtered_code_list = _get_filtered_code_list(filter, list_df, code_col)
    if filtered_code_list is not None:
        return _apply_filter(df, filtered_code_list, code_col)
    return df

def _get_filtered_code_list(filter: str, list_df: pd.DataFrame, code_col: str | None) -> list:
    '''絞り込み条件から、対象銘柄の銘柄コード一覧を取得します。'''
    return list_df.query(filter)[code_col].astype(str).unique().tolist()

def _apply_filter(df: pd.DataFrame, filtered_code_list: list[str], code_col: str | None) -> pd.DataFrame:
    '''絞り込みを適用します。'''
    return df[df[code_col].astype(str).isin(filtered_code_list)]