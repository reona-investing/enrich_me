from datetime import datetime
import pandas as pd
from sklearn.decomposition import PCA

class PCAHandler:
    @staticmethod
    def apply_PCA(df: pd.DataFrame,
                  n_components: int,
                  fit_start: datetime,
                  fit_end: datetime) -> pd.DataFrame:
        '''
        PCAを適用し、次元削減を行う。
        Args:
            df (pd.DataFrame): Target列を含む二階層インデックス付きデータフレーム
        Returns:
            pd.DataFrame: PCA適用後の価格情報
        '''
        if df.index.nlevels != 2:
            raise ValueError('インデックスは2階層にしてください。')

        df_for_pca, df_for_fit = PCAHandler._prepare_fit_and_transform_dfs(df, fit_start, fit_end)
        return PCAHandler._extract_pc(df_for_pca, df_for_fit, n_components)
    
    @staticmethod
    def get_residuals_from_PCA(df: pd.DataFrame,
                                n_components: int,
                                fit_start: datetime,
                                fit_end: datetime) -> pd.DataFrame:
        '''
        PCAを適用し、主成分を除去した残差を返す。
        Args:
            df (pd.DataFrame): Target列を含む二階層インデックス付きデータフレーム
        Returns:
            pd.DataFrame: PCA残差
        '''
        if df.index.nlevels != 2:
            raise ValueError('インデックスは2階層にしてください。')

        df_for_pca, df_for_fit = PCAHandler._prepare_fit_and_transform_dfs(df, fit_start, fit_end)
        extracted_df = PCAHandler._extract_pc(df_for_pca, df_for_fit, n_components)
        return PCAHandler._compute_residuals(df_for_pca, extracted_df)

    @staticmethod
    def _prepare_fit_and_transform_dfs(df: pd.DataFrame, fit_start: datetime, fit_end: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_for_pca = df['Target'].unstack(-1)
        df_for_pca = df_for_pca[df_for_pca.index >= fit_start]
        df_for_fit = df_for_pca[df_for_pca.index <= fit_end]
        return df_for_pca, df_for_fit

    @staticmethod
    def _extract_pc(df_for_pca: pd.DataFrame, df_for_fit: pd.DataFrame, n_components: int) -> pd.DataFrame:
        pca = PCA(n_components=n_components).fit(df_for_fit)
        extracted_array = pca.transform(df_for_pca)
        extracted_df = pd.DataFrame(extracted_array, index=df_for_pca.index, columns=[f'PC_{j:02}' for j in range(n_components)]).sort_index(ascending=True)
        inverse_df = pca.inverse_transform(extracted_df)
        inverse_df.columns = df_for_pca.columns
        return inverse_df

    @staticmethod
    def _compute_residuals(df_original: pd.DataFrame, extracted_df: pd.DataFrame) -> pd.DataFrame:
        residuals = (df_original - extracted_df).unstack()
        residuals_df = pd.DataFrame(residuals).reset_index()
        return residuals_df.rename(columns={0: 'Target'}).set_index(['Date', 'Sector'])