#%% モジュールのインポート
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime

class TargetCalculator:
    @staticmethod
    def daytime_return(df: pd.DataFrame) -> pd.DataFrame: # 日中生リターンの算出
        '''
        日中生リターンを算出する。
        Args:
            df (pd.DataFrame): 元データ（Open, Closeの各列が必須）
        '''
        df['Target'] = df['Close'] / df['Open'] - 1
        target_df = df[['Target']]
        target_df = TargetCalculator._sort_index(target_df)

        print('特徴量（日中リターン）の算出が完了しました。')

        return target_df
    
    @staticmethod
    def daytime_return_PCAresiduals(df:pd.DataFrame, # PCAで残差をとった日中リターンの算出
                                    reduce_components:int, train_start_day:datetime, train_end_day:datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        PCAで残差を取った日中リターンを算出する。
        Args:
            df (pd.DataFrame): 元データ（Open, Closeの各列が必須）
            reduce_components (int): 抽出する主成分の数
            train_start_day (datetime): PCAのfit開始日
            train_end_day (datetime): PCAのfit終了日
        Returns:
            pd.DataFrame 生の日中リターン
            pd.DataFrame PCA処理済みの日中リターン
        '''
        #日中リターンの算出
        raw_target_df = TargetCalculator.daytime_return(df)
        target_df = TargetCalculator.get_PCAresiduals(raw_target_df, reduce_components, train_start_day, train_end_day)
        target_df = TargetCalculator._sort_index(target_df)

        print('特徴量（日中リターン_PCA残差）の算出が完了しました。')

        return raw_target_df, target_df

    @staticmethod
    def get_PCAresiduals(df:pd.DataFrame, reduce_components:int, # PCAの残差を取る
                        train_start_day:datetime, train_end_day:datetime) -> pd.DataFrame:
        '''
        PCA残差を抽出（大きな主成分を除去）する。
        Args:
            df (pd.DataFrame): PCA処理前のデータフレーム
            reduce_components (int): 抽出する主成分の数
            train_start_day (datetime): PCAのfit開始日
            train_end_day (datetime): PCAのfit終了日
        Returns:
            pd.DataFrame: PCA残差を抽出したデータフレーム
        '''
        if df.index.nlevels != 2:
            raise ValueError('この目的変数を採用するには、元のdfのインデックスを"Date", "Sector"の二階層としてください。')
        df_for_pca, df_for_fit = TargetCalculator._get_dfs_for_pca(df, train_start_day, train_end_day)
        extracted_df = TargetCalculator._extract_pc(df_for_pca, df_for_fit, extract_components = reduce_components)
        return TargetCalculator._get_pca_residuals(df_for_pca, extracted_df)
        
    @staticmethod
    def _sort_index(target_df:pd.DataFrame) -> pd.DataFrame: # インデックスをキーにdfをソートする
        return target_df.sort_index()

    @staticmethod
    def _get_dfs_for_pca(df: pd.DataFrame, train_start_day: datetime, train_end_day: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        PCA用のデータフレームを作成します。
        Args:
            df (pd.DataFrame): PCA処理前のデータフレーム
            train_start_day (datetime): PCAのfit開始日
            train_end_day (datetime): PCAのfit終了日
        Returns:
            pd.DataFrame: PCA処理するデータフレーム
            pd.DataFrame: PCAのfit用データフレーム
        '''
        df_for_pca = df['Target'].unstack(-1)
        df_for_pca = df_for_pca[df_for_pca.index>=train_start_day]
        df_for_fit = df_for_pca[df_for_pca.index<=train_end_day]
        return df_for_pca, df_for_fit    

    @staticmethod
    def _extract_pc(df_for_pca: pd.DataFrame, df_for_fit: pd.DataFrame, extract_components: int) -> pd.DataFrame:
        '''
        PCAを適用し、指定した数の主成分を抽出します。
        Args:
            df_for_pca (pd.DataFrame): PCA処理するデータフレーム
            df_for_fit (pd.DataFrame): PCAのfit用データフレーム
            extract_components (int): 抽出する主成分の数
        Returns:
            pd.DataFrame: 指定した数の主成分を抽出したデータフレーム
        '''
        pca = PCA(n_components = extract_components).fit(df_for_fit)
        extracted_array = pca.transform(df_for_pca)
        extracted_df = pd.DataFrame(extracted_array, index=df_for_pca.index, columns=['PC_'+ '{:0=2}'.format(j) for j in range(0, extract_components)])
        extracted_df = extracted_df.sort_index(ascending=True)
        extracted_df = pca.inverse_transform(extracted_df)
        extracted_df.columns = df_for_pca.columns
        return extracted_df

    @staticmethod
    def _get_pca_residuals(df_for_pca: pd.DataFrame, extracted_df: pd.DataFrame) -> pd.DataFrame:
        '''
        PCA残差の抽出操作を行う。
        Args:
            df_for_pca (pd.DataFrame): PCA処理するデータフレーム
            extracted_df (pd.DataFrame): 指定した数の主成分を抽出したデータフレーム
        Returns:
            pd.DataFrame: PCA残差を抽出したデータフレーム
        '''
        residuals = (df_for_pca - extracted_df).unstack()
        residuals_df = pd.DataFrame(residuals).reset_index()
        return residuals_df.rename(columns={0:'Target'}).set_index(['Date', 'Sector'])



#%% デバッグ
if __name__ == '__main__':
    from utils.paths import Paths
    
    NEW_SECTOR_PRICE_PKLGZ = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/sector_price/New48sectors_price.parquet'
    df = pd.read_parquet(NEW_SECTOR_PRICE_PKLGZ)
    df = df.set_index(['Date', 'Sector'], drop=True)
    raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(df, 1, datetime(2016,1,1), datetime.today())
    print(target_df)
