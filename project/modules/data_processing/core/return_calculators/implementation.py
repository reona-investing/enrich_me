"""
リターン算出エンジンの具体実装
"""
import pandas as pd
from typing import Optional, List

from .base import ReturnCalculator, TimeSeriesReturnCalculator, GroupedReturnCalculator
from ..contracts.input_contracts import get_input_contract
from ..contracts.output_contracts import get_output_contract


class IntradayReturnCalculator(TimeSeriesReturnCalculator):
    """日中リターン算出（Close/Open - 1）"""
    
    def __init__(self, **kwargs):
        input_contract = get_input_contract('sector_price')  # マルチインデックス対応
        output_contract = get_output_contract('target', target_column='Target')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
    
    @property
    def output_column_name(self) -> str:
        return "Target"
    
    @property
    def required_input_columns(self) -> List[str]:
        return ["Open", "Close"]
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """日中リターンを算出"""
        result = df.copy()
        result[self.output_column_name] = df['Close'] / df['Open'] - 1
        return result[[self.output_column_name]]


class DailyReturnCalculator(TimeSeriesReturnCalculator):
    """日次リターン算出（前日比）"""
    
    def __init__(self, **kwargs):
        input_contract = get_input_contract('sector_price')  # マルチインデックス対応
        output_contract = get_output_contract('target', target_column='1d_return')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
    
    @property
    def output_column_name(self) -> str:
        return "1d_return"
    
    @property
    def required_input_columns(self) -> List[str]:
        return ["Close"]
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """日次リターンを算出"""
        result = df.copy()
        
        if df.index.nlevels > 1 and 'Sector' in df.index.names:
            # セクター別に計算
            result[self.output_column_name] = df.groupby(level='Sector')['Close'].pct_change(1)
        else:
            # 全体で計算
            result[self.output_column_name] = df['Close'].pct_change(1)
        
        return result[[self.output_column_name]]


class OvernightReturnCalculator(TimeSeriesReturnCalculator):
    """オーバーナイトリターン算出（Open/前日Close - 1）"""
    
    def __init__(self, **kwargs):
        input_contract = get_input_contract('sector_price')  # マルチインデックス対応
        output_contract = get_output_contract('target', target_column='overnight_return')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
    
    @property
    def output_column_name(self) -> str:
        return "overnight_return"
    
    @property
    def required_input_columns(self) -> List[str]:
        return ["Open", "Close"]
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """オーバーナイトリターンを算出"""
        result = df.copy()
        
        if df.index.nlevels > 1 and 'Sector' in df.index.names:
            # セクター別に計算
            prev_close = df.groupby(level='Sector')['Close'].shift(1)
        else:
            # 全体で計算
            prev_close = df['Close'].shift(1)
        
        result[self.output_column_name] = df['Open'] / prev_close - 1
        return result[[self.output_column_name]]


class BondReturnCalculator(TimeSeriesReturnCalculator):
    """債券リターン算出（価格差分）"""
    
    def __init__(self, **kwargs):
        input_contract = get_input_contract('index')
        output_contract = get_output_contract('target', target_column='bond_return')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
    
    @property
    def output_column_name(self) -> str:
        return "bond_return"
    
    @property
    def required_input_columns(self) -> List[str]:
        return ["Close"]
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """債券リターンを算出（価格差分）"""
        result = df.copy()
        
        if df.index.nlevels > 1:
            # グループ別に計算
            group_level = df.index.names[-1] if 'Sector' in df.index.names else df.index.names[1]
            result[self.output_column_name] = df.groupby(level=group_level)['Close'].diff(1)
        else:
            # 全体で計算
            result[self.output_column_name] = df['Close'].diff(1)
        
        return result[[self.output_column_name]]


class CommodityJPYReturnCalculator(TimeSeriesReturnCalculator):
    """コモディティ円建てリターン算出"""
    
    def __init__(self, usdjpy_data: Optional[pd.DataFrame] = None, **kwargs):
        input_contract = get_input_contract('index')
        output_contract = get_output_contract('target', target_column='commodity_jpy_return')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
        self.usdjpy_data = usdjpy_data
    
    @property
    def output_column_name(self) -> str:
        return "commodity_jpy_return"
    
    @property
    def required_input_columns(self) -> List[str]:
        return ["Close"]
    
    def calculate(self, df: pd.DataFrame, usdjpy_data: Optional[pd.DataFrame] = None, **kwargs) -> pd.DataFrame:
        """コモディティ円建てリターンを算出"""
        if usdjpy_data is None and self.usdjpy_data is None:
            raise ValueError("USDJPYデータが必要です")
        
        usdjpy = usdjpy_data if usdjpy_data is not None else self.usdjpy_data
        
        result = df.copy()
        
        # USDJPYデータを結合
        if 'Date' in df.index.names:
            # インデックスがDateを含む場合
            merged_df = pd.merge(df.reset_index(), usdjpy[['Close']].rename(columns={'Close': 'USDJPY'}), 
                               left_on='Date', right_index=True, how='left')
            merged_df = merged_df.set_index(df.index.names)
        else:
            # インデックスがDateでない場合
            merged_df = pd.merge(df, usdjpy[['Close']].rename(columns={'Close': 'USDJPY'}), 
                               left_index=True, right_index=True, how='left')
        
        # 円建て価格を計算してリターンを算出
        jpy_price = merged_df['Close'] * merged_df['USDJPY']
        
        if df.index.nlevels > 1:
            # グループ別に計算
            group_level = df.index.names[-1] if 'Sector' in df.index.names else df.index.names[1]
            result[self.output_column_name] = jpy_price.groupby(level=group_level).pct_change(1)
        else:
            # 全体で計算
            result[self.output_column_name] = jpy_price.pct_change(1)
        
        return result[[self.output_column_name]]


class CustomReturnCalculator(TimeSeriesReturnCalculator):
    """カスタムリターン算出（任意の関数を適用）"""
    
    def __init__(self, calc_function, output_column: str = 'custom_return', 
                 required_columns: List[str] = ['Close'], **kwargs):
        input_contract = get_input_contract('index')
        output_contract = get_output_contract('target', target_column=output_column)
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
        self.calc_function = calc_function
        self.output_column = output_column
        self.required_columns = required_columns
    
    @property
    def output_column_name(self) -> str:
        return self.output_column
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.required_columns
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """カスタム関数でリターンを算出"""
        result = df.copy()
        
        if df.index.nlevels > 1:
            # グループ別に計算
            group_level = df.index.names[-1] if 'Sector' in df.index.names else df.index.names[1]
            result[self.output_column_name] = df.groupby(level=group_level).apply(
                lambda x: self.calc_function(x, **kwargs)
            ).reset_index(level=0, drop=True)
        else:
            # 全体で計算
            result[self.output_column_name] = self.calc_function(df, **kwargs)
        
        return result[[self.output_column_name]]


class MultiPeriodReturnCalculator(GroupedReturnCalculator):
    """複数期間のリターンを同時計算"""
    
    def __init__(self, periods: List[int] = [1, 5, 21], return_type: str = 'daily', **kwargs):
        input_contract = get_input_contract('sector_price')  # マルチインデックス対応
        output_contract = get_output_contract('features')
        super().__init__(input_contract=input_contract, output_contract=output_contract, **kwargs)
        self.periods = periods
        self.return_type = return_type
    
    @property
    def output_column_name(self) -> str:
        return f"{self.return_type}_return_multi"
    
    @property
    def required_input_columns(self) -> List[str]:
        if self.return_type == 'intraday':
            return ["Open", "Close"]
        else:
            return ["Close"]
    
    def calculate(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """複数期間のリターンを算出"""
        result = pd.DataFrame(index=df.index)
        
        for period in self.periods:
            if self.return_type == 'intraday' and period == 1:
                # 1日の日中リターン
                result[f'{period}d_intraday_return'] = df['Close'] / df['Open'] - 1
            else:
                # 日次リターン
                if df.index.nlevels > 1 and 'Sector' in df.index.names:
                    # セクター別に計算
                    result[f'{period}d_return'] = df.groupby(level='Sector')['Close'].pct_change(period)
                else:
                    # 全体で計算
                    result[f'{period}d_return'] = df['Close'].pct_change(period)
        
        return result


# ファクトリー関数
def get_return_calculator(calculator_type: str, **kwargs) -> ReturnCalculator:
    """計算機タイプに応じたリターン計算機を取得"""
    calculators = {
        'intraday': IntradayReturnCalculator,
        'daily': DailyReturnCalculator,
        'overnight': OvernightReturnCalculator,
        'bond': BondReturnCalculator,
        'commodity_jpy': CommodityJPYReturnCalculator,
        'custom': CustomReturnCalculator,
        'multi_period': MultiPeriodReturnCalculator
    }
    
    if calculator_type not in calculators:
        raise ValueError(f"未対応の計算機タイプです: {calculator_type}")
    
    calculator_class = calculators[calculator_type]
    return calculator_class(**kwargs)


# 便利関数
def calculate_returns_batch(data_dict: dict, calculator_configs: dict) -> dict:
    """
    複数のデータに対してバッチでリターン計算を実行
    
    Args:
        data_dict: データ名をキーとするデータフレーム辞書
        calculator_configs: 計算機設定の辞書
        
    Returns:
        計算結果の辞書
    """
    results = {}
    
    for data_name, data_df in data_dict.items():
        if data_name in calculator_configs:
            config = calculator_configs[data_name]
            calculator = get_return_calculator(**config)
            results[data_name] = calculator.execute(data_df)
        else:
            # デフォルト設定で日次リターンを計算
            calculator = get_return_calculator('daily')
            results[data_name] = calculator.execute(data_df)
    
    return results