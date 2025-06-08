import pandas as pd
from sklearn.decomposition import PCA

class PCAResidualExtractor:
    """指定した主成分数を除去した残差を返す簡易クラス"""

    def __init__(self, remove_components: int = 0) -> None:
        self.remove_components = remove_components
        self.pca: PCA | None = None

    def fit(self, df: pd.DataFrame) -> 'PCAResidualExtractor':
        if self.remove_components <= 0:
            return self
        self.pca = PCA(n_components=self.remove_components)
        self.pca.fit(df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.remove_components <= 0 or self.pca is None:
            return df.copy()
        comp = self.pca.inverse_transform(self.pca.transform(df))
        residuals = df - comp
        return pd.DataFrame(residuals, index=df.index, columns=df.columns)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
