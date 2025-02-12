from utils.paths import Paths
from utils import yaml_utils
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Any



class SectorFinCalculator:
    def __init__(self, 
                 fin_yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML,
                 price_yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML,
                 sector_index_yaml_path: str = Paths.SECTOR_INDEX_COLUMNS_YAML,
                 sector_fin_yaml_path: str = Paths.SECTOR_FIN_COLUMNS_YAML):

        self.fin_yaml = yaml_utils.including_columns_loader(fin_yaml_path, 'original_columns') \
            + yaml_utils.including_columns_loader(fin_yaml_path, 'calculated_columns')
        self.price_yaml =  yaml_utils.including_columns_loader(price_yaml_path, 'original_columns')
        self.sector_index_yaml = yaml_utils.including_columns_loader(sector_index_yaml_path, 'calculated_columns')
        self.sector_fin_yaml = yaml_utils.including_columns_loader(sector_fin_yaml_path, 'calculated_columns')
    
    def _get_column_names(self):
        col = {'fin_date': self._column_name_getter(self.fin_yaml, 'DisclosedDate'),
               'fin_code': self._column_name_getter(self.fin_yaml, 'LocalCode'),
               'fin_forecast_eps': self._column_name_getter(self.fin_yaml, 'FORECAST_EPS'),
               'fin_outstanding_shares': self._column_name_getter(self.fin_yaml, 'OUTSTANDING_SHARES'),
               'price_date': self._column_name_getter(self.price_yaml, 'Date'),
               'price_code': self._column_name_getter(self.price_yaml, 'Code'),
               'price_close': self._column_name_getter(self.price_yaml, 'Close'),
               'sector': self._column_name_getter(self.sector_index_yaml, 'SECTOR'),
               'sector_fin_date': self._column_name_getter(self.sector_fin_yaml, 'DATE'),
               'sector_fin_sector': self._column_name_getter(self.sector_fin_yaml, 'SECTOR'),
               'sector_fin_market_cap': self._column_name_getter(self.sector_fin_yaml, 'MARKET_CAP'),
               'sector_fin_eps': self._column_name_getter(self.sector_fin_yaml, 'EPS'),
               }
        return col

    def calculate(self, SECTOR_FIN_PARQUET:str, fin_df: pd.DataFrame, price_df: pd.DataFrame, sector_dif_info: pd.DataFrame, 
                  sector_dif_info_code_col: str = 'Code'):
        '''
        Args:
            fin_df (pd.DataFrame): 財務情報
            price_df (pd.DataFrame): 価格情報
            sector_dif_info (pd.DataFrame): セクター
        '''
        col = self._get_column_names()

        fin_df[col['fin_code']] = fin_df[col['fin_code']].astype(str)
        fin_df[col['fin_date']] = pd.to_datetime(fin_df[col['fin_date']])
        price_df[col['price_code']] = price_df[col['price_code']].astype(str)
        price_df[col['price_date']] = pd.to_datetime(price_df[col['price_date']])
        sector_dif_info[sector_dif_info_code_col] = sector_dif_info[sector_dif_info_code_col].astype(str)
        
        price_df = price_df.rename(columns = {col['price_code']: col['fin_code'], col['price_date']: col['fin_date']})
        sector_dif_info = sector_dif_info.rename(columns = {sector_dif_info_code_col: col['fin_code']})
        
        merged_df = pd.merge(price_df, fin_df, how = 'outer', on = [col['fin_date'], col['fin_code']]) 
        merged_df = pd.merge(merged_df, sector_dif_info, how = 'left', on = col['fin_code'])
        
        cols_to_ffill = [x for x in merged_df.columns if x not in price_df.columns]

        merged_df[cols_to_ffill] = merged_df.groupby(col['fin_code'])[cols_to_ffill].ffill()

        merged_df[col['sector_fin_market_cap']] = merged_df[col['price_close']] * merged_df[col['fin_outstanding_shares']]   
        sector_fin_df = merged_df.groupby([col['price_date'], col['sector']])[[col['sector_fin_market_cap']]].mean()
        sector_fin_df[col['sector_fin_eps']] = merged_df.groupby([col['fin_date'], col['sector']]).apply(self._weighted_average, col['fin_forecast_eps'])

        sector_fin_df = sector_fin_df.rename(columns={col['fin_date']: col['sector_fin_date'], 
                                              col['sector']: col['sector_fin_sector']})
        sector_fin_df.to_parquet(SECTOR_FIN_PARQUET)

        return sector_fin_df

    def _weighted_average(self, group, group_name: str):
        col = self._get_column_names()
        d = group[col['sector_fin_market_cap']].sum()
        if d != 0:
            return (group[group_name] * group[col['sector_fin_market_cap']]).sum() / d
        else:
            return 0    
    # --------------------------------------------------------------------------
    #  以下、ヘルパーメソッド
    # --------------------------------------------------------------------------

    def _column_name_getter(self, yaml_info: dict[str | dict[str | Any]], raw_name: str) -> str:
        """
        指定したカラム名の変換後の名称を取得。

        Args:
            yaml_info (dict[str | dict[str | Any]])
            raw_name (str): 変換前のカラム名

        Returns:
            str: 変換後のカラム名
        """ 
        return yaml_utils.column_name_getter(yaml_info, {'name': raw_name}, 'fixed_name')


if __name__ == '__main__':
    from facades import StockAcquisitionFacade

    sector_dif_info = pd.read_csv(f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv')

    saf = StockAcquisitionFacade(filtered_code_list=sector_dif_info['Code'].astype(str))
    stock_dfs = saf.get_stock_data_dict()
    
    sfc = SectorFinCalculator()
    sector_fin_df = sfc.calculate(f'{Paths.SECTOR_FIN_FOLDER}/New48sectors_fin.parquet', stock_dfs['fin'], stock_dfs['price'], sector_dif_info)