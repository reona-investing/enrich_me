# utils/formatting.py
import pandas as pd

def format_contracts_df(df: pd.DataFrame, sector_list_df: pd.DataFrame):
    df['銘柄コード'] = df['銘柄コード'].astype(str)
    df['株数'] = df['株数'].astype(int)
    df['取得単価'] = df['取得単価'].astype(float)
    df['決済単価'] = df['決済単価'].astype(float)

    df['取得価格'] = (df['取得単価'] * df['株数']).astype(int)
    df['決済価格'] = (df['決済単価'] * df['株数']).astype(int)
    df['手数料'] = 0
    df['利益（税引前）'] = 0
    df.loc[df['売or買']=='買', '利益（税引前）'] = df['決済価格'] - df['取得価格'] - df['手数料']
    df.loc[df['売or買']=='売', '利益（税引前）'] = df['取得価格'] - df['決済価格'] - df['手数料']
    df['利率（税引前）'] = df['利益（税引前）'] / df['取得価格']

    sector_list_df['Code'] = sector_list_df['Code'].astype(str)
    df = pd.merge(df, sector_list_df[['Code', 'Sector']], left_on='銘柄コード', right_on='Code', how='left')
    df = df.drop('Code', axis=1).rename(columns={'Sector':'業種'})
    df = df[['日付', '売or買', '業種', '銘柄コード', '社名', '株数', '取得単価', '決済単価', '取得価格', '決済価格', '手数料', '利益（税引前）', '利率（税引前）']]
    return df

def format_cashflow_transactions_df(df: pd.DataFrame) -> pd.DataFrame:
    '''データフレームを'''
    #日付型に変換
    df['日付'] = pd.to_datetime(df['入出金日']).dt.date

    # ハイフンや空文字を0に変換して、数値型に変換
    for col in ["出金額", "入金額", "振替出金額", "振替入金額"]:
        df[col] = df[col].astype(str).replace("-", "0")
        df[col] = df[col].str.replace(",", "")
        df[col] = df[col].astype(int)


    df['入出金額'] = df['入金額'] + df['振替入金額'] - df['出金額'] - df['振替出金額']
    df = df.loc[~df['摘要'].str.contains('譲渡益税')]
    df = df[['日付', '摘要', '入出金額']]

    return df