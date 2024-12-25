import pandas as pd

def to_parquet(data: pd.DataFrame, path: str) -> None:
    """データを Parquet ファイルに保存."""
    data.to_parquet(path)
    print(f"データを保存しました: {path}")