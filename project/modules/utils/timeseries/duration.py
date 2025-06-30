from dataclasses import dataclass
from datetime import datetime
import pandas as pd

@dataclass
class Duration:
    start: datetime
    end: datetime

    def extract_from_df(self, df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """
        指定された期間でDataFrameをフィルタリングします。
        datetime_columnがインデックス、または列として存在する場合に対応します。
        """
        temp_df = df.copy() # オリジナルを変更しないためにコピー

        # datetime_columnがDataFrameのインデックス名に存在するかチェック
        if datetime_column in temp_df.index.names:
            # インデックスのdatetime_columnに対応するレベルの値を直接フィルタリング
            # マルチインデックスの場合を考慮して、該当するレベルを取得
            if isinstance(temp_df.index, pd.MultiIndex):
                # datetime_columnがどのレベルにあるか確認し、そのレベルでフィルタリング
                level_idx = temp_df.index.names.index(datetime_column)
                temp_df.index = temp_df.index.set_levels(pd.to_datetime(temp_df.index.levels[level_idx], errors='coerce'), level=level_idx)
                filtered_df = temp_df[(temp_df.index.get_level_values(datetime_column) >= self.start) & 
                                      (temp_df.index.get_level_values(datetime_column) <= self.end)]
            else:
                # シングルインデックスの場合
                temp_df.index = pd.to_datetime(temp_df.index, errors='coerce')
                filtered_df = temp_df[(temp_df.index >= self.start) & (temp_df.index <= self.end)]

            return filtered_df

        # datetime_columnがDataFrameの列に存在するかチェック
        elif datetime_column in temp_df.columns:
            # 列をdatetime型に変換
            temp_df[datetime_column] = pd.to_datetime(temp_df[datetime_column], errors='coerce')
            # 列でフィルタリング
            filtered_df = temp_df[(temp_df[datetime_column] >= self.start) & 
                                  (temp_df[datetime_column] <= self.end)]
            return filtered_df
        else:
            raise ValueError(f"'{datetime_column}' はDataFrameのインデックス名または列に存在しません。")