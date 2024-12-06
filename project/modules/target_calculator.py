#%% モジュールのインポート
from sklearn.decomposition import PCA
import pandas as pd
from datetime import datetime
import sklearn

#%% 関数群
def _sort_index(target_df:pd.DataFrame) -> pd.DataFrame: # インデックスをキーにdfをソートする
    target_df = target_df.sort_index()
    return target_df

def _get_PCAresiduals(df:pd.DataFrame, reduce_components:int, # PCAの残差を取る
                      train_start_day:datetime, train_end_day:datetime) -> pd.DataFrame:
    if df.index.nlevels != 2:
        raise ValueError('この目的変数を採用するには、元のdfのインデックスを"Date", "Sector"の二階層としてください。')

    #必要な各要素個別のデータフレームを作成する．
    df_for_pca = df['Target'].unstack(-1)
    df_for_pca = df_for_pca[df_for_pca.index>=train_start_day]
    df_train_for_pca = df_for_pca[df_for_pca.index<=train_end_day]

    #PCAで次元削減して、extract_dfとして残差を抽出
    pca = PCA(n_components=reduce_components).fit(df_train_for_pca)
    pca_array = pca.transform(df_for_pca)
    pca_df = pd.DataFrame(pca_array, index=df_for_pca.index, columns=['PC_'+ '{:0=2}'.format(j) for j in range(0, reduce_components)])
    pca_df = pca_df.sort_index(ascending=True)

    inversed_df = pca.inverse_transform(pca_df).copy()
    if sklearn.__version__ <= '1.2.2':
        inversed_df = pd.DataFrame(inversed_df, index=df_for_pca.index)
    inversed_df.columns = df_for_pca.columns
    extracted_df = df_for_pca - inversed_df

    #目的変数の作成
    extracted_df = extracted_df.unstack()
    target_df = pd.DataFrame(extracted_df).reset_index()
    target_df = target_df.rename(columns={0:'Target'}).set_index(['Date', 'Sector'])

    return target_df


def daytime_return(df: pd.DataFrame) -> pd.DataFrame: # 日中生リターンの算出
    '''日中生リターンの算出'''
    df['Target'] = df['Close'] / df['Open'] - 1
    target_df = df[['Target']]
    target_df = _sort_index(target_df)

    print('特徴量（日中リターン）の算出が完了しました。')

    return target_df

def daytime_return_PCAresiduals(df:pd.DataFrame, # PCAで残差をとった日中リターンの算出
                                reduce_components:int, train_start_day:datetime, train_end_day:datetime) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    PCAで残差を取った日中リターンの算出
    返り値1: raw_target_df 生の日中リターン
    返り値2: target_df PCA処理済みの日中リターン
    '''
    #日中リターンの算出
    raw_target_df = daytime_return(df)
    target_df = _get_PCAresiduals(raw_target_df, reduce_components, train_start_day, train_end_day)
    target_df = _sort_index(target_df)

    print('特徴量（日中リターン_PCA残差）の算出が完了しました。')

    return raw_target_df, target_df

#%% デバッグ
if __name__ == '__main__':
    import paths
    
    from IPython.display import display
    NEW_SECTOR_PRICE_PKLGZ = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price.pkl.gz'
    df = pd.read_parquet(NEW_SECTOR_PRICE_PKLGZ)
    df = df.set_index(['Date', 'Sector'], drop=True)
    raw_target_df, target_df = daytime_return_PCAresiduals(df, 1, datetime(2016,1,1), datetime.today())
    display(target_df)
