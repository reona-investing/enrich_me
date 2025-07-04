from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict
import pandas as pd


@dataclass
class MLOutputCollection:
    """機械学習の出力データ群を管理するクラス"""

    raw_returns_df: pd.DataFrame = field(repr=False)
    pred_result_df: pd.DataFrame = field(repr=False)
    order_price_df: pd.DataFrame = field(repr=False)

    def update_outputs(
        self,
        new_raw_returns: Optional[pd.DataFrame] = None,
        new_pred_result: Optional[pd.DataFrame] = None,
        new_order_price: Optional[pd.DataFrame] = None,
    ) -> "MLOutputCollection":
        """出力データを更新したインスタンスを返す。"""
        return MLOutputCollection(
            raw_returns_df=new_raw_returns.sort_index() if new_raw_returns is not None else self.raw_returns_df,
            pred_result_df=new_pred_result.sort_index() if new_pred_result is not None else self.pred_result_df,
            order_price_df=new_order_price.sort_index() if new_order_price is not None else self.order_price_df,
        )


class MLOutputCollectionStorage:
    """MLOutputCollectionインスタンスのセーブ・ロードを司るクラス"""

    _RAW_RETURN_FILE = "raw_returns_df.parquet"
    _PRED_RESULT_FILE = "pred_result_df.parquet"
    _ORDER_PRICE_FILE = "order_price_df.parquet"

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Dict[str, Path]:
        return {
            "raw_returns_df": self.base_path / self._RAW_RETURN_FILE,
            "pred_result_df": self.base_path / self._PRED_RESULT_FILE,
            "order_price_df": self.base_path / self._ORDER_PRICE_FILE,
        }

    def load(self) -> MLOutputCollection:
        """外部ファイルをロードしてMLOutputCollectionを作成します。"""
        return MLOutputCollection(
            raw_returns_df=pd.read_parquet(self.path["raw_returns_df"]),
            pred_result_df=pd.read_parquet(self.path["pred_result_df"]),
            order_price_df=pd.read_parquet(self.path["order_price_df"]),
        )

    def save(self, output_collection: MLOutputCollection) -> None:
        """MLOutputCollectionのプロパティを外部ファイルに出力します。"""
        self._atomic_write_parquet(output_collection.raw_returns_df, self.path["raw_returns_df"])
        self._atomic_write_parquet(output_collection.pred_result_df, self.path["pred_result_df"])
        self._atomic_write_parquet(output_collection.order_price_df, self.path["order_price_df"])

    @staticmethod
    def _atomic_write_parquet(obj: pd.DataFrame, dest: Path, compression: str = "zstd") -> None:
        if obj is not None:
            tmp_path = dest.with_suffix(dest.suffix + ".tmp")
            obj.to_parquet(tmp_path, compression=compression)
            tmp_path.replace(dest)