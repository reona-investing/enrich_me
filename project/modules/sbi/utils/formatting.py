# utils/formatting.py
import pandas as pd

def format_deal_result_df(df: pd.DataFrame, sector_list_df: pd.DataFrame):
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

def format_in_out_df(table):
    data = []
    for tr in table.find("tbody").findAll("tr"):
        row = []
        row.append(tr.findAll("td")[0].getText().strip()) # 日付
        row.append(tr.findAll("td")[2].getText().strip()) # 摘要
        row.append(tr.findAll("td")[3].getText().replace("-", "0").replace(",", "").strip()) # 出金額
        row.append(tr.findAll("td")[4].getText().replace("-", "0").replace(",", "").strip()) # 入金額
        row.append(tr.findAll("td")[5].getText().replace("-", "0").replace(",", "").strip()) # 振替出金額
        row.append(tr.findAll("td")[6].getText().replace("-", "0").replace(",", "").strip()) # 振替入金額
        data.append(row)

    columns = ["日付", "摘要", "出金額", "入金額", "振替入金額", "振替出金額"]
    in_out_df = pd.DataFrame(data, columns=columns)
    in_out_df['日付'] = pd.to_datetime(in_out_df['日付']).dt.date
    for x in ['入金額', '出金額', '振替入金額', '振替出金額']:
        in_out_df[x] = in_out_df[x].astype(int)
    in_out_df['入出金額'] = in_out_df['入金額'] + in_out_df['振替入金額'] - in_out_df['出金額'] - in_out_df['振替出金額']
    in_out_df = in_out_df.loc[~in_out_df['摘要'].str.contains('譲渡益税')]
    in_out_df = in_out_df[['日付', '摘要', '入出金額']]
    return in_out_df