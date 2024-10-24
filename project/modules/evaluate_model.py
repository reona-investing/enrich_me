#%% モジュールのインポート
import paths
import data_pickler
import MLDataset
import calculate_stats

from typing import Tuple
import pickle
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, norm, t, chi2
import inspect
from datetime import datetime
import ipywidgets as widgets
from IPython.display import display, clear_output

#%% 関数
def calculate_return_topix(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    '''コントロールとしてTOPIXのリターンを算出'''
    #データ読み込み
    df = pd.read_csv(paths.FEATURES_TO_SCRAPE_CSV)
    topix_folder = df.loc[df['Name']=='TOPIX', 'Group'].values[0]
    topix_pkl = df.loc[df['Name']=='TOPIX', 'Path'].values[0]
    topix_path = paths.SCRAPED_DATA_FOLDER + '/' + topix_folder + '/' + topix_pkl
    topix_df = pd.read_parquet(topix_path)
    #リターン算出
    return_topix = pd.DataFrame()
    return_topix['Date'] = pd.to_datetime(topix_df['Date'])
    return_topix['TOPIX_daytime'] = topix_df['Close'] / topix_df['Open'] - 1
    return_topix['TOPIX_hold'] = topix_df['Close'] / topix_df['Open'].iat[0]
    return_topix['TOPIX_hold'] = return_topix['TOPIX_hold'].pct_change(1)
    return_topix = return_topix[(return_topix['Date'] >= start_date) & (return_topix['Date'] <= end_date)]
    return return_topix.set_index('Date')

#%% クラス
class LongShortModel:
    '''
    Long-Short戦略のモデルの予測結果を格納し、各種指標を計算します。
    **kwargsの説明
    bin_num: int 分位リターンのbin数。指定しない場合はsector数が割り当てられる。
    sector_to_estimate_num: int 上位（下位）何sectorを取引の対象とするか。
    top_float: 最上位（最下位）予想業種にどれだけの比率で傾斜を掛けるか？
    '''
    def __init__(self, model_name: str, pred_result_df: pd.DataFrame, raw_target_df: pd.DataFrame, start_date: datetime, end_date: datetime, **kwargs):
        # データフレームを格納する辞書を初期化
        self.dataframes_dict = {}
        # 各列のランクを取得
        self.df = pred_result_df.loc[(start_date <= pred_result_df.index.get_level_values('Date')) & \
                                     (pred_result_df.index.get_level_values('Date') <= end_date)].copy()
        raw_target_df = raw_target_df.loc[(start_date <= raw_target_df.index.get_level_values('Date')) & \
                                     (raw_target_df.index.get_level_values('Date') <= end_date)].copy()
        self.df['Target'] = raw_target_df['Target']
        self.original_columns = pred_result_df.columns
        for column in self.original_columns:
            self.df[f'{column}_Rank'] = self.df.groupby('Date')[column].rank()
        # 各関数を実行
        self.get_return_by_bin(**kwargs)
        self.get_longshort_df(**kwargs)
        self.get_longshort_probability()
        self.get_longshort_subTOPIX(start_date, end_date)
        self.get_monthly_longshort()
        self.get_daily_sector_performances(**kwargs)
        self.get_monthly_sector_performances()
        self.get_total_sector_performances()
        self.get_spearman_corr()
        self.get_numerai_corrs()


        #基本データの出力
        print(f'{model_name}の読み込みが完了しました。')
        sharpe_ratio = self.dataframes_dict['日次成績（集計）'].loc["SR", "LS"]
        print('全期間')
        estimated_df = calculate_stats.estimate_population(self.dataframes_dict['日次成績']['LS'], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print('直近1年')
        estimated_1y_df = calculate_stats.estimate_population(self.dataframes_dict['日次成績']['LS'].iloc[-252:], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print('直近3ヶ月')
        estimated_1y_df = calculate_stats.estimate_population(self.dataframes_dict['日次成績']['LS'].iloc[-63:], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print(f'最大ドローダウン (実績)： {round(self.dataframes_dict["日次成績"]["MaxDD"].max() * 100, 3)}%, {self.dataframes_dict["日次成績"]["MaxDDdays"].max()}日')
        print(f'Spearman順位相関： {round(self.dataframes_dict["Spearman相関（集計）"].loc["mean", "SpearmanCorr"], 5)}')
        print(f'Numerai相関： {round(self.dataframes_dict["Numerai相関（集計）"].loc["mean", "NumeraiCorr"], 5)}')
        print(f'----------------------------------------------------------------------------------')


    def get_return_by_bin(self, bin_num: int=None, **kwargs) -> pd.DataFrame:
        '''
        指定したbin数ごとのリターンを算出。
        bin_num：bin数。指定しなかった場合は、Sectorの数が自動で割り当てられる。
        '''
        df = self.df.dropna().copy()
        if bin_num is None:
            bin_num = int(df[['Target_Rank']].max().iloc[0] + 1)
        columns = df.columns
        for column in self.original_columns:
            df[f'{column}_Bin'] = \
              df.groupby('Date')[f'{column}_Rank'].transform(lambda x: pd.qcut(x, q=bin_num, labels=False))
        self.dataframes_dict['分位成績'] = df.groupby(['Date', 'Pred_Bin'])[['Target']].mean().unstack(-1)
        self.dataframes_dict['分位成績（集計）'] = self.dataframes_dict['分位成績'].stack(future_stack=True).groupby('Pred_Bin').describe().T


    def get_longshort_df(self, sector_to_extracted_num:int = 3, top_slope:float = 1, **kwargs) -> pd.DataFrame:
        '''
        Long, Short, LSの各リターンを算出。
        sector_to_extracted_num：抽出する業種数
        '''
        df = self.df.copy()
        columns = df.columns
        short_df = - df.loc[df['Pred_Rank'] <= sector_to_extracted_num]
        short_df.loc[short_df['Pred_Rank'] == short_df['Pred_Rank'].min(), 'Target'] *= top_slope
        short_df.loc[short_df['Pred_Rank'] != short_df['Pred_Rank'].min(), 'Target'] *= (1 - (top_slope - 1) / (sector_to_extracted_num - 1))
        short_df = short_df.groupby('Date')[['Target']].mean()
        short_df = short_df.rename(columns={'Target':'Short'})
        long_df = df.loc[df['Pred_Rank'] > max(df['Pred_Rank']) - sector_to_extracted_num]
        long_df.loc[long_df['Pred_Rank'] == long_df['Pred_Rank'].max(), 'Target'] *= top_slope
        long_df.loc[long_df['Pred_Rank'] != long_df['Pred_Rank'].max(), 'Target'] *= (1 - (top_slope - 1) / (sector_to_extracted_num - 1))
        long_df = long_df.groupby('Date')[['Target']].mean()
        long_df = long_df.rename(columns={'Target':'Long'})
        longshort_df = pd.concat([long_df, short_df], axis=1)
        longshort_df['LS'] = (longshort_df['Long'] + longshort_df['Short']) / 2

        longshort_agg = longshort_df.describe()
        longshort_agg.loc['SR'] = \
          longshort_agg.loc['mean'] / longshort_agg.loc['std']

        longshort_df['L_Cumprod'] = (longshort_df['Long'] + 1).cumprod() - 1
        longshort_df['S_Cumprod'] = (longshort_df['Short'] + 1).cumprod() - 1
        longshort_df['LS_Cumprod'] = (longshort_df['LS'] + 1).cumprod() - 1
        longshort_df['DD'] = 1 - (longshort_df['LS_Cumprod'] + 1) / (longshort_df['LS_Cumprod'].cummax() + 1)
        longshort_df['MaxDD'] = longshort_df['DD'].cummax()
        longshort_df['DDdays'] = longshort_df.groupby((longshort_df['DD'] == 0).cumsum()).cumcount()
        longshort_df['MaxDDdays'] = longshort_df['DDdays'].cummax()
        longshort_df = longshort_df[['LS', 'LS_Cumprod',
                                     'DD', 'MaxDD', 'DDdays', 'MaxDDdays',
                                     'Long', 'Short',	'L_Cumprod', 'S_Cumprod']]
        self.dataframes_dict['日次成績'] = longshort_df
        self.dataframes_dict['日次成績（集計）'] = longshort_agg


    def get_longshort_probability(self) -> pd.DataFrame:
        '''
        Long, Short, LSの各リターンを算出。
        sector_to_extracted_num：抽出する業種数
        '''
        longshort_df = self.dataframes_dict['日次成績'][['LS']].copy()
        longshort_df['CumMean'] = longshort_df['LS'].expanding().mean().shift(1)
        longshort_df['CumStd'] = longshort_df['LS'].expanding().std().shift(1)
        longshort_df['DegFreedom'] = (longshort_df['LS'].expanding().count() - 1).shift(1)
        longshort_df = longshort_df.dropna(axis=0)
        # 前日までのリターンの結果で、その時点での母平均と母標準偏差の信頼区間を算出
        longshort_df['MeanTuple'] = \
          longshort_df.apply(lambda row: calculate_stats.estimate_population_mean(row['CumMean'], row['CumStd'], CI=0.997, deg_freedom=row['DegFreedom']), axis=1)
        longshort_df['StdTuple'] = \
          longshort_df.apply(lambda row: calculate_stats.estimate_population_std(row['CumStd'], CI=0.997, deg_freedom=row['DegFreedom']), axis=1)
        longshort_df[['MeanWorst', 'MeanBest']] = longshort_df['MeanTuple'].apply(pd.Series)
        longshort_df[['StdWorst', 'StdBest']] = longshort_df['StdTuple'].apply(pd.Series)
        # 当日のリターンがどの程度の割合で起こるのかを算出
        longshort_df['Probability'] = round(longshort_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['CumMean'], scale=row['CumStd'])), axis=1), 6)
        longshort_df['ProbabWorst'] = round(longshort_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['MeanWorst'], scale=row['StdWorst'])), axis=1), 6)
        longshort_df['ProbabBest'] = round(longshort_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['MeanBest'], scale=row['StdBest'])), axis=1), 6)
        longshort_df = longshort_df[['LS', 'Probability', 'ProbabWorst', 'ProbabBest']]
        self.dataframes_dict['日次成績（確率分布）'] = longshort_df


    def get_longshort_subTOPIX(self, start_date: datetime, end_date: datetime):
        '''
        '''
        return_topix = calculate_return_topix(start_date, end_date)
        longshort_df = pd.merge(self.dataframes_dict['日次成績'], return_topix[['TOPIX_daytime']], how='left', left_index=True, right_index=True)
        longshort_df['Long'] -= longshort_df['TOPIX_daytime']
        longshort_df['Short'] += longshort_df['TOPIX_daytime']

        longshort_agg = longshort_df[['Long', 'Short', 'LS']].describe()
        longshort_agg.loc['SR'] = \
          longshort_agg.loc['mean'] / longshort_agg.loc['std']

        longshort_df['L_Cumprod'] = (longshort_df['Long'] + 1).cumprod() - 1
        longshort_df['S_Cumprod'] = (longshort_df['Short'] + 1).cumprod() - 1
        longshort_df = longshort_df[['LS', 'LS_Cumprod',
                                     'DD', 'MaxDD', 'DDdays', 'MaxDDdays',
                                     'Long', 'Short',	'L_Cumprod', 'S_Cumprod']]
        self.dataframes_dict['日次成績（TOPIX差分）'] = longshort_df
        self.dataframes_dict['日次成績（集計・TOPIX差分）'] = longshort_agg


    def get_monthly_longshort(self):
        monthly_longshort = (self.dataframes_dict['日次成績'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort_subTOPIX = (self.dataframes_dict['日次成績（TOPIX差分）'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort['Cumprod'] = (monthly_longshort['LS'] + 1).cumprod() - 1
        monthly_longshort_subTOPIX['Cumprod'] = (monthly_longshort_subTOPIX['LS'] + 1).cumprod() - 1
        self.dataframes_dict['月次成績'] = monthly_longshort
        self.dataframes_dict['月次成績（TOPIX差分）'] = monthly_longshort_subTOPIX


    def get_daily_sector_performances(self, sector_to_extracted_num: int=3, **kwargs):
        #業種ごとの成績
        #longとshortを返す
        df = self.df.copy()
        long_theshold = len(df.index.get_level_values('Sector').unique()) - sector_to_extracted_num + 1
        self.long_sectors = df[df['Pred_Rank'] >= long_theshold]
        self.short_sectors = df[df['Pred_Rank'] <= sector_to_extracted_num]
        #日次リターンの算出
        sector_performances_daily = pd.concat([self.long_sectors, self.short_sectors], axis=0)
        sector_performances_daily = sector_performances_daily.reset_index().sort_values(['Date', 'Pred_Rank'], ascending=True).set_index(['Date', 'Sector'], drop=True)
        self.dataframes_dict['セクター別成績（日次）'] = sector_performances_daily.copy()

    def get_monthly_sector_performances(self):
        #月次リターンの算出
        long_sectors = self.long_sectors.copy()
        short_sectors = self.short_sectors.copy()
        long_sectors = long_sectors.loc[long_sectors.index.get_level_values('Date') != long_sectors.index.get_level_values('Date')[-1]]
        short_sectors = short_sectors.loc[short_sectors.index.get_level_values('Date') != short_sectors.index.get_level_values('Date')[-1]]
        long_sectors['Target'] += 1
        short_sectors['Target'] -= 1
        short_sectors['Target'] = short_sectors['Target'].abs()
        long_grouped  = long_sectors.groupby([pd.Grouper(level='Date', freq='ME'), 'Sector'])
        short_grouped = short_sectors.groupby([pd.Grouper(level='Date', freq='ME'), 'Sector'])
        # ①月ごとのデータの件数
        long_count = pd.DataFrame(long_grouped.size(), columns=['Long_Num'])
        short_count = pd.DataFrame(short_grouped.size(), columns=['Short_Num'])
        # ②月ごとの'Target'列のすべての値の積
        long_return = long_grouped[['Target']].apply(np.prod)
        short_return = short_grouped[['Target']].apply(np.prod)

        long_monthly = pd.concat([long_count, long_return], axis=1)
        short_monthly = pd.concat([short_count, short_return], axis=1)
        monthly_sector_performances = pd.merge(long_monthly, short_monthly, how='outer', left_index=True, right_index=True).fillna(0)
        self.dataframes_dict['セクター別成績（月次）'] = monthly_sector_performances.copy()

    def get_total_sector_performances(self):
        #全期間トータル
        long_sectors = self.long_sectors.groupby('Sector')[['Target_Rank']].describe().droplevel(0, axis=1)
        long_sectors = long_sectors[['count', 'mean', '50%', 'std']]
        long_sectors.columns = ['Long_num', 'Long_mean', 'Long_med', 'Long_std']
        short_sectors = self.short_sectors.groupby('Sector')[['Target_Rank']].describe().droplevel(0, axis=1)
        short_sectors = short_sectors[['count', 'mean', '50%', 'std']]
        short_sectors.columns = ['Short_num', 'Short_mean', 'Short_med', 'Short_std']

        #topとbottomを結合して1つのdfに
        sector_performances_df = pd.merge(long_sectors, short_sectors, how='outer', left_index=True, right_index=True).fillna(0)
        sector_performances_df['Total_num'] = sector_performances_df['Long_num'] + sector_performances_df['Short_num']
        self.dataframes_dict['セクター別成績（トータル）'] = sector_performances_df.sort_values('Total_num', ascending=False)


    '''以下、相関係数関係'''
    def get_spearman_corr(self):
        '''
      日次のsperamanの順位相関係数と，その平均や標準偏差を算出
        '''
        daily_spearman = [spearmanr(x['Target_Rank'], x['Pred_Rank'])[0] for _, x in self.df.groupby('Date')]
        dateindex = self.df.index.get_level_values('Date').unique()
        self.dataframes_dict['Spearman相関'] = pd.DataFrame(daily_spearman, index=dateindex, columns=['SpearmanCorr'])
        self.dataframes_dict['Spearman相関（集計）'] = self.dataframes_dict['Spearman相関'].describe()


    def _get_daily_numerai_corr(self, target: pd.Series, pred_rank: pd.Series):
        '''
        日次のnumerai_corrを算出
        '''
        #pred_rankの前処理
        pred_rank = np.array(pred_rank)
        scaled_pred_rank = (pred_rank - 0.5) / len(pred_rank)
        gauss_pred_rank = norm.ppf(scaled_pred_rank)
        pred_p15 = np.sign(gauss_pred_rank) * np.abs(gauss_pred_rank) ** 1.5

        #targetの前処理
        target = np.array(target)
        centered_target = target - target.mean()
        target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5

        return np.corrcoef(pred_p15, target_p15)[0, 1]


    def _get_daily_numerai_rank_corr(self, target_rank: pd.Series, pred_rank: pd.Series):
        '''
        日次のnumerai_corr（Targetをランク化）を算出
        '''
        #Target, Predのそれぞれに前処理を施しリストに格納
        processed_ranks = []
        for x in [target_rank, pred_rank]:
            x = np.array(x)
            scaled_x = (x - 0.5) / len(x)
            gauss_x = norm.ppf(scaled_x)
            x_p15 = np.sign(gauss_x) * np.abs(gauss_x) ** 1.5
            processed_ranks.append(x_p15)

        return np.corrcoef(processed_ranks[0], processed_ranks[1])[0, 1]


    def get_numerai_corrs(self):
        '''
        日次のnumerai_corrと，その平均や標準偏差を算出
        numerai_corr：上位・下位の予測に重みづけされた，より実践的な指標．
        '''
        #Target順位化の有無（RawReturnとの相関も算出したい．））
        daily_numerai = self.df.groupby('Date').apply(lambda x: self._get_daily_numerai_corr(x['Target_Rank'], x['Pred_Rank']))
        daily_numerai_rank = self.df.groupby('Date').apply(lambda x: self._get_daily_numerai_rank_corr(x['Target_Rank'], x['Pred_Rank']))
        #データフレーム化
        dateindex = self.df.index.get_level_values('Date').unique()
        self.dataframes_dict['Numerai相関'] = pd.concat([daily_numerai, daily_numerai_rank], axis=1)
        self.dataframes_dict['Numerai相関'].columns = ['NumeraiCorr', 'Rank_NumeraiCorr']
        self.dataframes_dict['Numerai相関（集計）'] = self.dataframes_dict['Numerai相関'].describe()


    def display_result(self):
        '''任意のデータフレームを動的に描画'''
        #ウィジェットの構成要素の定義
        dropdown = widgets.Dropdown(
            options=self.dataframes_dict.keys(),
            description='選択：'
          )
        button = widgets.Button(description="表示")
        output = widgets.Output()
        #ボタンクリック時の挙動を定義
        def on_button_click(sb):
            selected_df = self.dataframes_dict[dropdown.value]
            with output:
                pd.set_option('display.max_rows', None)  # 表示行数を無制限に設定
                pd.set_option('display.max_columns', None)  # 表示列数を無制限に設定
                output.clear_output()
                display(selected_df)
                pd.reset_option('display.max_rows')  # 表示行数をデフォルトに戻す
                pd.reset_option('display.max_columns')  # 表示列数をデフォルトに戻す
        button.on_click(on_button_click)
        #ウィジェットの表示
        display(widgets.HBox([dropdown, button]))
        display(output)

#%% デバッグ
if __name__ == '__main__':
    ML_DATASET_PATH = f'{paths.ML_DATASETS_FOLDER}/New48sectors.pkl.gz'
    ml_dataset = data_pickler.load_from_records(ML_DATASET_PATH)
    model_stats = LongShortModel('48Sector', ml_dataset.pred_result_df, ml_dataset.raw_target_df, 
                                 start_date=datetime(2022,1,1), end_date=datetime.today(), bin_num=5, top_slope=1.5)
    model_stats.display_result()