from datetime import datetime
import pandas as pd
from preprocessing.pca_handler import PCAHandler

class TargetCalculator:
    @staticmethod
    def daytime_return(df: pd.DataFrame) -> pd.DataFrame:
        '''
        日中生リターンを算出する。
        Args:
            df (pd.DataFrame): 元データ（Open, Closeの各列が必須）
        Returns:
            pd.DataFrame: 日中リターン（Target列）を含むDataFrame
        '''
        df['Target'] = df['Close'] / df['Open'] - 1
        target_df = df[['Target']]
        return target_df.sort_index()

    @staticmethod
    def daytime_return_PCAresiduals(df: pd.DataFrame,
                                     reduce_components: int,
                                     train_start_day: datetime,
                                     train_end_day: datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
        '''
        PCAで主成分を除去した日中リターンの残差を算出。
        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: 生の日中リターン, PCA残差リターン
        '''
        raw_target_df = TargetCalculator.daytime_return(df)
        target_df = PCAHandler.get_residuals_from_PCA(raw_target_df, reduce_components, train_start_day, train_end_day).sort_index(ascending=True)
        return raw_target_df, target_df

#%% デバッグ
if __name__ == '__main__':
    from utils.paths import Paths
    
    NEW_SECTOR_PRICE_PKLGZ = f'{Paths.SECTOR_REDEFINITIONS_FOLDER}/sector_price/New48sectors_price.parquet'
    df = pd.read_parquet(NEW_SECTOR_PRICE_PKLGZ)
    df = df.set_index(['Date', 'Sector'], drop=True)
    raw_target_df, target_df = TargetCalculator.daytime_return_PCAresiduals(df, 1, datetime(2016,1,1), datetime.today())
    print(target_df.tail(5))
