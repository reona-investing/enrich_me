import pandas as pd
  

def filter_stocks(df:pd.DataFrame, list_df: pd.DataFrame, filter:str, filtered_code_list:list) -> pd.DataFrame: # 対象銘柄の抜き取り
    '''
    対象銘柄を抜き取ります。
    df: 抜き取り対象のデータフレーム
    filter: 銘柄の絞り込み条件をstr型で指定
    filtered_code_list: 絞り込み対象の銘柄をリスト型で指定
    filter と filtered_code_listはどちらかを入力する。
    どちらも入力されている場合、filterが優先される
    '''
    if filter:
        filtered_code_list = _get_filtered_code_list(filter, list_df)
    if filtered_code_list is not None:
        return df[df['Code'].astype(str).isin(filtered_code_list)]
    return df

def _get_filtered_code_list(filter: str, list_df: pd.DataFrame) -> list:
    return list_df.query(filter)['Code'].astype(str).unique()  