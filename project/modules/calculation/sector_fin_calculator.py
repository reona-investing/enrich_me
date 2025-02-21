from utils.paths import Paths
from utils.yaml_utils import ColumnConfigsGetter
from utils import yaml_utils
import pandas as pd
from typing import Any



class SectorFinCalculator:
    def __init__(self, 
                 fin_yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML,
                 price_yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML,
                 sector_index_yaml_path: str = Paths.SECTOR_INDEX_COLUMNS_YAML,
                 sector_fin_yaml_path: str = Paths.SECTOR_FIN_COLUMNS_YAML):
        
        self.fin_col_configs = ColumnConfigsGetter(fin_yaml_path)
        self.price_col_configs = ColumnConfigsGetter(price_yaml_path)
        self.sector_index_col_configs = ColumnConfigsGetter(sector_index_yaml_path)
        self.sector_fin_col_configs = ColumnConfigsGetter(sector_fin_yaml_path)
    
    def _get_column_names(self):
        fin_col = self.fin_col_configs.get_all_columns_name_asdict()
        price_col = self.price_col_configs.get_all_columns_name_asdict()
        sector_index_col = self.sector_index_col_configs.get_all_columns_name_asdict()
        sector_fin_col = self.sector_fin_col_configs.get_all_columns_name_asdict()

        return fin_col, price_col, sector_index_col, sector_fin_col

    def calculate(self, SECTOR_FIN_PARQUET:str, fin_df: pd.DataFrame, price_df: pd.DataFrame, sector_dif_info: pd.DataFrame, 
                  sector_dif_info_code_col: str = 'Code'):
        '''
        Args:
            fin_df (pd.DataFrame): 財務情報
            price_df (pd.DataFrame): 価格情報
            sector_dif_info (pd.DataFrame): セクター
        '''
        fin_col, price_col, sector_index_col, sector_fin_col = self._get_column_names()

        fin_df[fin_col['銘柄コード']] = fin_df[fin_col['銘柄コード']].astype(str)
        fin_df[fin_col['日付']] = pd.to_datetime(fin_df[fin_col['日付']])
        price_df[price_col['銘柄コード']] = price_df[price_col['銘柄コード']].astype(str)
        price_df[price_col['日付']] = pd.to_datetime(price_df[price_col['日付']])
        sector_dif_info[sector_dif_info_code_col] = sector_dif_info[sector_dif_info_code_col].astype(str)
        
        price_df = price_df.rename(columns = {price_col['銘柄コード']: fin_col['銘柄コード'], price_col['日付']: fin_col['日付']})
        sector_dif_info = sector_dif_info.rename(columns = {sector_dif_info_code_col: fin_col['銘柄コード']})
        
        merged_df = pd.merge(price_df, fin_df, how = 'outer', on = [fin_col['日付'], fin_col['銘柄コード']]) 
        merged_df = pd.merge(merged_df, sector_dif_info, how = 'left', on = fin_col['銘柄コード'])
        
        cols_to_ffill = [x for x in merged_df.columns if x not in price_df.columns]

        merged_df[cols_to_ffill] = merged_df.groupby(fin_col['銘柄コード'])[cols_to_ffill].ffill()

        merged_df[sector_fin_col['時価総額']] = merged_df[price_col['終値']] * merged_df[fin_col['発行済み株式数']]   
        sector_fin_df = merged_df.groupby([price_col['日付'], sector_index_col['セクター']])[[sector_fin_col['時価総額']]].mean()
        sector_fin_df[sector_fin_col['予想EPS']] = \
            merged_df.groupby([fin_col['日付'], sector_index_col['セクター']]).apply(self._weighted_average, fin_col['予想EPS'])

        sector_fin_df = sector_fin_df.rename(columns={fin_col['日付']: sector_fin_col['日付'], 
                                              sector_index_col['セクター']: sector_fin_col['セクター']})
        sector_fin_df.to_parquet(SECTOR_FIN_PARQUET)

        return sector_fin_df

    def _weighted_average(self, group, group_name: str):
        _, _, _, sector_fin_col = self._get_column_names()
        d = group[sector_fin_col['時価総額']].sum()
        if d != 0:
            return (group[group_name] * group[sector_fin_col['時価総額']]).sum() / d
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
        return yaml_utils.column_configs_getter(yaml_info, {'name': raw_name}, 'fixed_name')


if __name__ == '__main__':
    from facades import StockAcquisitionFacade
    sector_dif_info = pd.read_csv(f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/48sectors_2024-2025.csv')

    saf = StockAcquisitionFacade(filtered_code_list=sector_dif_info['Code'].astype(str))
    stock_dfs = saf.get_stock_data_dict()
    
    sfc = SectorFinCalculator()
    sector_fin_df = sfc.calculate(f'{Paths.SECTOR_FIN_FOLDER}/New48sectors_fin.parquet', stock_dfs['fin'], stock_dfs['price'], sector_dif_info)
    print(sector_fin_df)