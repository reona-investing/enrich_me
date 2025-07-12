import pandas as pd

from utils.jquants_api_utils import get_next_open_date
from utils.timeseries import Duration


class TrainTestData:
    """
    訓練・テストデータの管理と前処理を担当
    外部からは読み取り専用プロパティだけを公開し、
    内部状態は保護する。
    """

    # 1) 「このクラスには決められた属性しか生やさない」宣言
    __slots__ = (
        "_target_train_df", "_target_test_df",
        "_features_train_df", "_features_test_df"
    )

    # 2) 再代入を禁止
    def __setattr__(self, name, value):
        if name in self.__slots__ and hasattr(self, name):
            raise AttributeError(f"{name} is read-only")
        super().__setattr__(name, value)

    # ---------- 公開 API ----------
    # 読み取り専用プロパティ（コピーを返す）
    @property
    def target_train_df(self) -> pd.DataFrame:
        return self._target_train_df.copy(deep=True)

    @property
    def target_test_df(self) -> pd.DataFrame:
        return self._target_test_df.copy(deep=True)

    @property
    def features_train_df(self) -> pd.DataFrame:
        return self._features_train_df.copy(deep=True)

    @property
    def features_test_df(self) -> pd.DataFrame:
        return self._features_test_df.copy(deep=True)

    def archive(
        self,
        target_df: pd.DataFrame,
        features_df: pd.DataFrame,
        train_duration: Duration,
        test_duration: Duration,
        datetime_column: str,
        outlier_threshold: float = 0,
        no_shift_features: list[str] = [],
        reuse_features_df: bool = False
    ) -> 'TrainTestData':
        """データの前処理と分割を実行"""
        
        # 翌営業日を追加
        target_df = self._append_next_business_day_row(target_df)
        if not reuse_features_df:
            features_df = self._append_next_business_day_row(features_df)
        
        target_df = target_df[~target_df.index.duplicated(keep='first')]
        features_df = features_df[~features_df.index.duplicated(keep='first')]

        features_df = self._shift_features(features_df, no_shift_features)
        features_df = self._align_index(features_df, target_df)

        # 学習・テストデータに分割
        target_train = train_duration.extract_from_df(df=target_df, datetime_column=datetime_column)
        features_train = train_duration.extract_from_df(df=features_df, datetime_column=datetime_column)
        target_test= test_duration.extract_from_df(df=target_df, datetime_column=datetime_column)
        features_test = test_duration.extract_from_df(df=features_df, datetime_column=datetime_column)
        
        # 外れ値除去
        if outlier_threshold != 0:
            target_train, features_train = \
                self._remove_outliers(target_train, features_train, outlier_threshold=outlier_threshold)
        object.__setattr__(self, "_target_train_df", target_train)
        object.__setattr__(self, "_target_test_df", target_test)
        object.__setattr__(self, "_features_train_df", features_train)
        object.__setattr__(self, "_features_test_df", features_test)

        return self


    def _remove_outliers(self, target_train: pd.DataFrame, features_train: pd.DataFrame, outlier_threshold: float):
        """外れ値除去"""
        target_train, features_train = self._filter_outliers_from_datasets(
            target_train, features_train, outlier_threshold
        )
        return target_train, features_train

    # === データ処理のコアメソッド群 ===
    
    def _append_next_business_day_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """次の営業日の行を追加"""
        next_open_date = get_next_open_date(latest_date=df.index.get_level_values('Date')[-1])
        sectors = df.index.get_level_values('Sector').unique()
        new_rows = [[next_open_date for _ in range(len(sectors))], [sector for sector in sectors]]

        data_to_add = pd.DataFrame(index=new_rows, columns=df.columns).dropna(axis=1, how='all')
        data_to_add.index.names = ['Date', 'Sector']

        df = pd.concat([df, data_to_add], axis=0).reset_index(drop=False)
        df['Date'] = pd.to_datetime(df['Date'])
        return df.set_index(['Date', 'Sector'], drop=True)

    def _shift_features(self, features_df: pd.DataFrame, no_shift_features: list) -> pd.DataFrame:
        """特徴量を1日シフト"""
        shift_features = [col for col in features_df.columns if col not in no_shift_features]
        features_df[shift_features] = features_df.groupby('Sector')[shift_features].shift(1)
        return features_df

    def _align_index(self, features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """特徴量のインデックスを目的変数と揃える（共通部分だけにする）"""
        common_index = features_df.index.intersection(target_df.index)
        return features_df.loc[common_index]

    def _filter_outliers_from_datasets(self, target_train: pd.DataFrame, features_train: pd.DataFrame,
                                      outlier_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """外れ値除去の実行"""
        target_train = target_train.groupby('Sector').apply(
            self._filter_outliers_by_group, column_name='Target', coef=outlier_threshold
        ).droplevel(0, axis=0)
        target_train = target_train.sort_index()
        features_train = features_train.loc[features_train.index.isin(target_train.index), :]
        return target_train, features_train

    def _filter_outliers_by_group(self, group: pd.DataFrame, column_name: str, coef: float = 3) -> pd.DataFrame:
        """グループ単位での外れ値除去"""
        mean = group[column_name].mean()
        std = group[column_name].std()
        lower_bound = mean - coef * std
        upper_bound = mean + coef * std
        return group[(group[column_name] >= lower_bound) & (group[column_name] <= upper_bound)]