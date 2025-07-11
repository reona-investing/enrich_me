import pandas as pd
from timeseries_data.public import return_timeseries
from timeseries_data.public.return_timeseries import ReturnTimeseries


class ReturnTimeseriesCollection:
    """複数のReturnTimeseriesインスタンスを格納するクラス"""

    def __init__(self):
        self._collection: list[ReturnTimeseries] = []

    def append(self, return_timeseries: ReturnTimeseries):
        """
        ReturnTimeseriesインスタンスを追加します。
        
        Args:
            return_timeseries (ReturnTimeseries): 追加するインスタンス
        """
        if return_timeseries.name in self.names:
            raise ValueError(f"{return_timeseries.name}インスタンスはすでに登録済みです。登録されるインスタンスの名称は一意である必要があります。")
        self._collection.append(return_timeseries)

    def clear(self):
        """
        格納したReturnTimeseriesインスタンスをすべて削除します。
        """
        self._collection = []

    def get_return_timeseries_instance(self, instance_name: str) -> ReturnTimeseries:
        """
        指定した名前のインスタンスを取り出します。
       
        Args:
            instance_name (str): 取り出したいインスタンスの名前
        """
        if self.length == 0:
            raise ValueError(f"まだインスタンスが1つも登録されていません。")
        instance_list = [instance for instance in self._collection if instance.name == instance_name]
        if len(instance_list):
            raise ValueError(f"{instance_name}インスタンスは存在しません。")
        return instance_list[0]

    def get_all_return_timeseries_instances(self) -> list[ReturnTimeseries]:
        """
        格納しているすべてのReturnTimeseriesインスタンスをリスト形式で取得します。
        """
        if self.length == 0:
            return []
        return [instance for instance in self._collection]
    
    @property
    def length(self) -> int:
        """格納しているReturnTimeseriesインスタンスの個数を返します。"""
        return len(self._collection)
    
    @property
    def names(self) -> list[str]:
        """格納しているReturnTimeseriesインスタンスの名称をリストとして返します。"""
        if self.length == 0:
            return []
        return [return_timeseries.name for return_timeseries in self._collection]

    @property
    def raw_merged_df(self) -> pd.DataFrame:
        merged_df = pd.DataFrame()
        if self.length == 0:
            return merged_df
        for return_timeseries in self._collection:
            merged_df = pd.merge(merged_df, return_timeseries.raw_return, how='outer', left_index=True, right_index=True)
        return merged_df

    @property
    def processed_merged_df(self) -> pd.DataFrame:
        merged_df = pd.DataFrame()
        if self.length == 0:
            return merged_df
        for return_timeseries in self._collection:
            merged_df = pd.merge(merged_df, return_timeseries.processed_return, how='outer', left_index=True, right_index=True)
        return merged_df