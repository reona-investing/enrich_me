import pandas as pd

class Formatter:
    @staticmethod
    def format_stock_code(code_column: pd.Series) -> pd.Series:
        '''普通株の銘柄コードを4桁に変換するヘルパー関数'''
        return code_column.str[:4].where((code_column.str.len() == 5)&(code_column.str[-1] == '0'), code_column)