import pandas as pd

class Formatter:
    @staticmethod
    def format_stock_code(df: pd.DataFrame) -> pd.DataFrame:
        '''普通株の銘柄コードを4桁に変換するヘルパー関数'''
        df["Code"] = \
            df["Code"].str[:4].where((df["Code"].str.len() == 5)&(df["Code"].str[-1] == '0'), df["Code"])
        return df