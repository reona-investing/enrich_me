import pandas as pd
from timeseries_data.calculation_method import CalculationMethodBase


class StockReturnTimeseries:
    def __init__(self, original_timeseries: pd.DataFrame, date_column: str = 'Date', sector_column: str = 'Sector'):
        self.original_timeseries = original_timeseries.copy()
        self.date_column = date_column
        self.sector_column = sector_column
        self._raw_return = None
        self._processed_return = None

    def calculate(self, method: CalculationMethodBase, *args, **kwargs):
        """
        指定されたメソッドでリターンを計算します。
        
        Args:
            method: 計算メソッドのインスタンス
            *args, **kwargs: calculateメソッドに渡す追加引数
        """
        original_index = self.original_timeseries.index.names
        original_timeseries = self.original_timeseries.reset_index(drop=False)
        self._raw_return = \
            original_timeseries.groupby(self.sector_column).apply(
                lambda group: method.calculate(group, *args, **kwargs)
                ).reset_index(drop=True).set_index(original_index)
        self._processed_return = self._raw_return.copy() # processed_returnは初期状態ではraw_returnと同じ
    
    def preprocess(self, pipeline):
        """
        前処理パイプラインを実行します。
        
        Args:
            pipeline (list): 前処理クラスのインスタンスのリスト
        """
        if self._processed_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        
        for processor in pipeline:
            self._processed_return = processor.calculate(self._processed_return)
    
    @property
    def raw_return(self) -> pd.DataFrame:
        """生のリターンデータを返します。"""
        if self._raw_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._raw_return.copy()
    
    @property
    def processed_return(self) -> pd.DataFrame:
        """前処理済みのリターンデータを返します。"""
        if self._processed_return is None:
            raise ValueError("calculate()メソッドを先に実行してください。")
        return self._processed_return.copy()

    def evaluate(self):
        """評価メソッド（将来の実装用）"""
        pass