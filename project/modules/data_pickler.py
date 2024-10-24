#%% モジュールのインポート
import MLDataset
from typing import Union
import numpy as np
import pandas as pd
import pickle
import gzip

#%% ヘルパー関数
def _prepare_index_for_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのインデックス列をリセットし、変換に備える。"""
    df_copy = df.copy()
    if df_copy.index.nlevels == 1:
        df_copy.index.name = 'Index_' + (df_copy.index.name if df_copy.index.name else '')
    else:
        df_copy.index.names = ['Index_' + name if name else '' for name in df_copy.index.names]
    return df_copy.reset_index(drop=False)

def _restore_index_after_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrameのインデックス列を復元する。"""
    index_cols = [col for col in df.columns if col.startswith('Index_')]
    df = df.set_index(index_cols, drop=True)
    if df.index.nlevels == 1:
        df.index.name = df.index.name.replace('Index_', '')
    else:
        df.index.names = [name.replace('Index_', '') for name in df.index.names]
    return df

#%% メインの関数
def _convert_df_to_records(df: pd.DataFrame) -> np.recarray:
    """pd.DataFrameをnp.recarrayに変換する"""
    df_copy = _prepare_index_for_conversion(df)
    column_dtypes = {col: dtype for col, dtype in df_copy.dtypes.items()}
    return df_copy.to_records(index=False, column_dtypes=column_dtypes)

def _convert_records_to_df(records: np.recarray) -> pd.DataFrame:
    """np.recarrayをpd.DataFrameに変換する"""
    df = pd.DataFrame(records)
    df = _restore_index_after_conversion(df)
    return df.loc[:, ~df.columns.str.contains('Unnamed')]

def _pickle_data(data: Union[np.recarray, 'MLDataset.MLDataset'], path: str) -> None:
    """データをpickleで保存する"""
    with gzip.open(path, 'wb', compresslevel=1) if path.endswith('.pkl.gz') else open(path, 'wb') as f:
        pickle.dump(data, f)

def _unpickle_data(path: str) -> Union[np.recarray, 'MLDataset.MLDataset']:
    """pickleファイルを読み込み、データを復元する"""
    with gzip.open(path, 'rb') if path.endswith('.pkl.gz') else open(path, 'rb') as f:
        return pickle.load(f)

def dump_as_records(data: Union[pd.DataFrame, 'MLDataset.MLDataset'], path: str) -> None:
    """
    pd.DataFrameをnp.recarrayに変換して保存する。
    MLDatasetの場合は内部のpd.DataFrameオブジェクトも変換する。
    """
    if isinstance(data, pd.DataFrame):
        records = _convert_df_to_records(data)
    elif isinstance(data, MLDataset.MLDataset):
        for attr, value in vars(data).items():
            if isinstance(value, pd.DataFrame):
                setattr(data, attr, _convert_df_to_records(value))
        records = data
    else:
        raise TypeError("data must be a pd.DataFrame or an MLDataset.MLDataset")

    _pickle_data(records, path)

def load_from_records(path: str) -> Union[pd.DataFrame, 'MLDataset.MLDataset']:
    """
    np.recarrayからpd.DataFrameに復元する。
    MLDatasetの場合は内部のnp.recarrayオブジェクトも復元する。
    """
    data = _unpickle_data(path)

    if isinstance(data, np.recarray):
        return _convert_records_to_df(data)
    elif isinstance(data, MLDataset.MLDataset):
        for attr, value in vars(data).items():
            if isinstance(value, np.recarray):
                setattr(data, attr, _convert_records_to_df(value))
        return data
    else:
        raise TypeError("Loaded data must be a np.recarray or an MLDataset.MLDataset")

#%% デバッグ
if __name__ == '__main__':
    import paths
    from IPython.display import display

    file_path = paths.RAW_STOCK_LIST_PARQUET
    data = load_from_records(file_path)
    dump_as_records(data, file_path)
