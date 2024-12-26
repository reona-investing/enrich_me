import os
import pandas as pd

class FileHandler:
    @staticmethod
    def file_exists(path: str) -> bool:
        return os.path.isfile(path)

    @staticmethod
    def read_parquet(path: str) -> pd.DataFrame:
        return pd.read_parquet(path)

    @staticmethod
    def write_parquet(data: pd.DataFrame, path: str, verbose=True) -> None:
        data.to_parquet(path)
        if verbose == True:
            print(f"データを保存しました: {path}")