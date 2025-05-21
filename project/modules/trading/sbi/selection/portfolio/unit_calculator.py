import pandas as pd
from typing import List
from trading.sbi.selection.interface import OrderUnit
from trading.sbi_trading_logic.price_limit_calculator import PriceLimitCalculator

class UnitCalculator:
    """注文単位計算クラス"""
    
    def __init__(self):
        self.price_limit_calculator = PriceLimitCalculator()
    
    def calculate_units(self, 
                       df: pd.DataFrame, 
                       max_cost: float,
                       direction: str) -> List[OrderUnit]:
        """注文単位数を計算"""
        # 理想コストを計算
        df = self._get_ideal_costs(df, max_cost)
        
        # 仮ポートフォリオを作成
        df = self._draft_portfolio(df)
        
        # 最大単位数に制限
        if 'MaxUnit' in df.columns:
            df = self._reduce_to_max_unit(df)
        
        # 予算内に抑える
        df = self._reduce_units(df, max_cost)
        
        # 予算を最大限活用
        df = self._increase_units(df, max_cost)
        
        # 不要なカラムを削除
        result_df = df.drop(['MaxUnitWithinIdeal', 'MaxCostWithinIdeal', 
                            'MinCostExceedingIdeal', 'ReductionRate'], 
                           axis=1, errors='ignore')
        
        # 価格上限を計算
        result_df = self._calculate_price_limits(result_df)
        
        # OrderUnitオブジェクトのリストに変換
        order_units = []
        for _, row in result_df[result_df['Unit'] > 0].iterrows():
            order_units.append(
                OrderUnit(
                    Code=row['Code'],
                    CompanyName=row.get('CompanyName', ''),
                    Sector=row['Sector'],
                    Unit=int(row['Unit']),
                    EstimatedCost=float(row['EstimatedCost']),
                    TotalCost=float(row['TotalCost']),
                    UpperLimitCost=float(row['UpperLimitCost']),
                    UpperLimitTotal=float(row['UpperLimitTotal']),
                    Direction=direction,
                    isBorrowingStock=bool(row.get('isBorrowingStock', False))
                )
            )
        
        return order_units
    
    def _get_ideal_costs(self, df: pd.DataFrame, max_cost: float) -> pd.DataFrame:
        """理想コストを計算"""
        df = df.sort_values('Weight', ascending=False).reset_index(drop=True).copy()
        df['IdealCost'] = (df['Weight'] * max_cost).astype(int)
        df['MaxUnitWithinIdeal'] = df['IdealCost'] // df['EstimatedCost']
        df['MaxCostWithinIdeal'] = df['EstimatedCost'] * df['MaxUnitWithinIdeal']
        df['MinCostExceedingIdeal'] = df['MaxCostWithinIdeal'] + df['EstimatedCost']
        return df
    
    def _draft_portfolio(self, df: pd.DataFrame) -> pd.DataFrame:
        """仮ポートフォリオを作成"""
        df['ReductionRate'] = abs(df['IdealCost'] - df['MaxCostWithinIdeal']) / abs(df['IdealCost'] - df['MinCostExceedingIdeal'])
        
        for index, row in df.iterrows():
            if row['ReductionRate'] <= 1:
                df.loc[index, 'TotalCost'] = row['MaxCostWithinIdeal']
            else:
                df.loc[index, 'TotalCost'] = row['MinCostExceedingIdeal']
            df.loc[index, 'Unit'] = df.loc[index, 'TotalCost'] / row['EstimatedCost']
        return df
    
    def _reduce_to_max_unit(self, df: pd.DataFrame) -> pd.DataFrame:
        """最大単位数に制限"""
        df = df.copy()
        for i in range(len(df)):
            index_to_reduce = df.index[i]
            if df.loc[index_to_reduce, 'Unit'] >= df.loc[index_to_reduce, 'MaxUnit']:
                df.loc[index_to_reduce, 'Unit'] = df.loc[index_to_reduce, 'MaxUnit']
                df.loc[index_to_reduce, 'TotalCost'] = df.loc[index_to_reduce, 'Unit'] * df.loc[index_to_reduce, 'EstimatedCost']
                df.loc[index_to_reduce, 'isMaxUnit'] = True
        return df
    
    def _reduce_units(self, df: pd.DataFrame, max_cost: float) -> pd.DataFrame:
        """予算内に抑える"""
        df = df.copy()
        candidate_indices = df.loc[(df['ReductionRate'] >= 1) & (df['Unit'] > 0), :].sort_values('ReductionRate', ascending=False).index
        
        for index_to_reduce in candidate_indices:
            if df['TotalCost'].sum() <= max_cost:
                break
            df.loc[index_to_reduce, 'Unit'] -= 1
            df.loc[index_to_reduce, 'TotalCost'] = df.loc[index_to_reduce, 'Unit'] * df.loc[index_to_reduce, 'EstimatedCost']
        
        return df
    
    def _increase_units(self, df: pd.DataFrame, max_cost: float) -> pd.DataFrame:
        """予算を最大限活用"""
        df = df.copy()
        while True:
            free_cost = max_cost - df['TotalCost'].sum()
            
            if 'isMaxUnit' in df.columns:
                filtered_df = df[(df['EstimatedCost'] <= free_cost) & (~df['isMaxUnit'])]
            else:
                filtered_df = df[df['EstimatedCost'] <= free_cost]
            
            if filtered_df.empty:
                break
            
            min_rate_row = filtered_df['ReductionRate'].idxmin()
            df.loc[min_rate_row, 'Unit'] += 1
            df.loc[min_rate_row, 'TotalCost'] = df.loc[min_rate_row, 'Unit'] * df.loc[min_rate_row, 'EstimatedCost']
            
            df.loc[min_rate_row, 'ReductionRate'] = abs(df.loc[min_rate_row, 'IdealCost'] - df.loc[min_rate_row, 'TotalCost']) / \
                abs(df.loc[min_rate_row, 'IdealCost'] - (df.loc[min_rate_row, 'TotalCost'] + df.loc[min_rate_row, 'EstimatedCost']))
            
            if 'MaxUnit' in df.columns and df.loc[min_rate_row, 'Unit'] >= df.loc[min_rate_row, 'MaxUnit']:
                df.loc[min_rate_row, 'Unit'] = df.loc[min_rate_row, 'MaxUnit']
                df.loc[min_rate_row, 'TotalCost'] = df.loc[min_rate_row, 'Unit'] * df.loc[min_rate_row, 'EstimatedCost']
                df.loc[min_rate_row, 'isMaxUnit'] = True
        
        return df
    
    def _calculate_price_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格上限を計算"""
        df = df.copy()
        df['UpperLimitCost'] = df['EstimatedCost'] / 100
        df['UpperLimitCost'] = df['UpperLimitCost'].apply(self.price_limit_calculator.calculate_upper_limit) * 100
        df['UpperLimitTotal'] = df['UpperLimitCost'] * df['Unit']
        return df