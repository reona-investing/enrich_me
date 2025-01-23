import pandas as pd
from utils.paths import Paths
from typing import Literal
from models import MLDataset
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr, norm
from models.evaluation import calculate_stats
import ipywidgets as widgets
from IPython.display import display

class DatetimeManager:
    def __init__(self, start_date: datetime, end_date: datetime):
        '''
        対象日を管理するサブモジュール
        Args:
            start_date (datetime): 評価開始日
            end_date (datetim): 評価終了日
        '''
        self.start_date = start_date
        self.end_date = end_date
        pass
    def extract_duration(self, df: pd.DataFrame):
        '''
        指定した日付のデータを抽出する。
        '''
        return df.loc[(self.start_date <= df.index.get_level_values('Date')) & (df.index.get_level_values('Date') <= self.end_date)]


class LSControlHandler:
    """
    コントロールデータを読み込む。
    type (Literal[str]): コントロールの種類を指定。デフォルトはTOPIX。
    """
    def __init__(self, type: Literal['TOPIX',] = 'TOPIX'):
        self.control_type = type
        if type == 'TOPIX':
            self.control_df = self._load_topix_data()
        self.control_df = self._calculate_return(self.control_df)

    def _load_topix_data(self) -> pd.DataFrame:
        """TOPIXデータをpandas.DataFrame形式で読み込む。"""
        topix_path = self._get_topix_path()
        return pd.read_parquet(topix_path)
    
    def _get_topix_path(self) -> str:
        '''TOPIXデータのファイルパスを取得する。'''
        df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
        topix_folder = df.loc[df['Name']=='TOPIX', 'Group'].values[0]
        topix_pkl = df.loc[df['Name']=='TOPIX', 'Path'].values[0]
        return Paths.SCRAPED_DATA_FOLDER + '/' + topix_folder + '/' + topix_pkl

    def _calculate_return(self,  control_df: pd.DataFrame) -> pd.DataFrame:
        '''リターンを計算する。'''
        df = pd.DataFrame()
        df['Date'] = pd.to_datetime(control_df['Date'])
        df[f'{self.control_type}_daytime'] = control_df['Close'] / control_df['Open'] - 1
        df[f'{self.control_type}_hold'] = (control_df['Close'] / control_df['Open'].iat[0]).pct_change(1)
        return df.set_index('Date', drop=True)


class LSDataHandler:
    """
    コントロールデータを読み込む
    type (Literal[str]): コントロールの種類を指定。デフォルトはTOPIX。
    """
    def __init__(self, 
                 ml_dataset: MLDataset, 
                 start_day: datetime = datetime(2013, 1, 1),
                 end_day: datetime = datetime.today(),
                 ls_control: LSControlHandler = LSControlHandler(),
                 ):
        self.datetime_manager = DatetimeManager(start_day, end_day)
        pred_result_df = self.datetime_manager.extract_duration(ml_dataset.pred_result_df)
        raw_target_df = self.datetime_manager.extract_duration(ml_dataset.raw_target_df)
        self.control_df = self.datetime_manager.extract_duration(ls_control.control_df)
        self.control_type = ls_control.control_type
        self.original_cols = pred_result_df.columns
        pred_result_df = self._append_rank_cols(pred_result_df)
        self.result_df = self._append_raw_target_col(pred_result_df, raw_target_df)

    def _append_rank_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        '''データフレームの各列をランク化します。'''
        df = df.copy()
        for column in df.columns:
            df[f'{column}_Rank'] = df.groupby('Date')[column].rank()
        return df

    def _append_raw_target_col(self, df: pd.DataFrame, raw_target_df: pd.DataFrame) -> pd.DataFrame:
        df.loc[:, 'RawTarget'] = raw_target_df['Target']
        return df


class MetricsCalculator:
    """指標計算を担当"""
    def __init__(self, 
                 ls_data_handler: LSDataHandler, 
                 trade_sectors_num: int = 3,
                 bin_num: int = None,
                 top_slope: float = 1):
        '''
        Long-Short戦略のモデルの予測結果を格納し、各種指標を計算します。
        Args:
            trade_sectors_num (int): 上位（下位）何sectorを取引の対象とするか。
            bin_num (int):分位リターンのbin数。指定しない場合はsector数が割り当てられる。
            top_slope (float): 最上位（最下位）予想業種にどれだけの比率で傾斜を掛けるか？
        '''
        self.result_df = ls_data_handler.result_df
        self.control_df = ls_data_handler.control_df
        self.control_type = ls_data_handler.control_type
        self.trade_sectors_num = trade_sectors_num
        if bin_num is None:
            bin_num = int(self.result_df[['RawTarget_Rank']].max().iloc[0] + 1)
        self.bin_num = bin_num
        self.top_slope = top_slope
        self.original_cols = ls_data_handler.original_cols
        self.metrics_functions = {}
        self.metrics_dfs = {}
        self._calculate_all_metrics()
    
    def _calculate_all_metrics(self):
        self.get_return_by_bin()
        self.get_longshort_df()
        self.get_longshort_probability()
        self.get_longshort_sub_control()
        self.get_monthly_longshort()
        self.get_daily_sector_performances()
        self.get_monthly_sector_performances()
        self.get_total_sector_performances()
        self.get_spearman_corr()
        self.get_numerai_corrs()


    def get_return_by_bin(self) -> pd.DataFrame:
        '''指定したbin数ごとのリターンを算出。'''
        df = self.result_df.dropna().copy()
        for column in self.original_cols:
            df[f'{column}_Bin'] = \
              df.groupby('Date')[f'{column}_Rank'].transform(lambda x: pd.qcut(x, q=self.bin_num, labels=False))
        self.metrics_dfs['分位成績'] = df.groupby(['Date', 'Pred_Bin'])[['Target']].mean().unstack(-1)
        self.metrics_dfs['分位成績（集計）'] = self.metrics_dfs['分位成績'].stack(future_stack=True).groupby('Pred_Bin').describe().T


    def get_longshort_df(self) -> pd.DataFrame:
        '''ロング・ショートそれぞれの結果を算出する。'''
        df = self.result_df.copy()
        short_df = - df.loc[df['Pred_Rank'] <= self.trade_sectors_num]
        short_df.loc[short_df['Pred_Rank'] == short_df['Pred_Rank'].min(), 'Target'] *= self.top_slope
        short_df.loc[short_df['Pred_Rank'] != short_df['Pred_Rank'].min(), 'Target'] *= (1 - (self.top_slope - 1) / (self.trade_sectors_num - 1))
        short_df = short_df.groupby('Date')[['Target']].mean()
        short_df = short_df.rename(columns={'Target':'Short'})
        long_df = df.loc[df['Pred_Rank'] > max(df['Pred_Rank']) - self.trade_sectors_num]
        long_df.loc[long_df['Pred_Rank'] == long_df['Pred_Rank'].max(), 'Target'] *= self.top_slope
        long_df.loc[long_df['Pred_Rank'] != long_df['Pred_Rank'].max(), 'Target'] *= (1 - (self.top_slope - 1) / (self.trade_sectors_num - 1))
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
        self.metrics_dfs['日次成績'] = longshort_df
        self.metrics_dfs['日次成績（集計）'] = longshort_agg


    def get_longshort_probability(self) -> pd.DataFrame:
        '''Long, Short, LSの各リターンを算出する。'''
        longshort_df = self.metrics_dfs['日次成績'][['LS']].copy()
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
        self.metrics_dfs['日次成績（確率分布）'] = longshort_df


    def get_longshort_sub_control(self):
        '''コントロールを差し引いたリターン結果を表示する。'''
        longshort_df = pd.merge(self.metrics_dfs['日次成績'], self.control_df[[f'{self.control_type}_daytime']], how='left', left_index=True, right_index=True)
        longshort_df['Long'] -= longshort_df[f'{self.control_type}_daytime']
        longshort_df['Short'] += longshort_df[f'{self.control_type}_daytime']

        longshort_agg = longshort_df[['Long', 'Short', 'LS']].describe()
        longshort_agg.loc['SR'] = \
          longshort_agg.loc['mean'] / longshort_agg.loc['std']

        longshort_df['L_Cumprod'] = (longshort_df['Long'] + 1).cumprod() - 1
        longshort_df['S_Cumprod'] = (longshort_df['Short'] + 1).cumprod() - 1
        longshort_df = longshort_df[['LS', 'LS_Cumprod',
                                     'DD', 'MaxDD', 'DDdays', 'MaxDDdays',
                                     'Long', 'Short',	'L_Cumprod', 'S_Cumprod']]
        self.metrics_dfs[f'日次成績（{self.control_type}差分）'] = longshort_df
        self.metrics_dfs[f'日次成績（集計・{self.control_type}差分）'] = longshort_agg


    def get_monthly_longshort(self):
        '''月次のリターンを算出'''
        monthly_longshort = (self.metrics_dfs['日次成績'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort_sub_control = (self.metrics_dfs[f'日次成績（{self.control_type}差分）'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort['Cumprod'] = (monthly_longshort['LS'] + 1).cumprod() - 1
        monthly_longshort_sub_control['Cumprod'] = (monthly_longshort_sub_control['LS'] + 1).cumprod() - 1
        self.metrics_dfs['月次成績'] = monthly_longshort
        self.metrics_dfs[f'月次成績（{self.control_type}差分）'] = monthly_longshort_sub_control


    def get_daily_sector_performances(self):
        '''業種ごとの成績を算出する'''
        df = self.result_df.copy()
        long_theshold = len(df.index.get_level_values('Sector').unique()) - self.trade_sectors_num + 1
        self.long_sectors = df[df['Pred_Rank'] >= long_theshold]
        self.short_sectors = df[df['Pred_Rank'] <= self.trade_sectors_num]
        #日次リターンの算出
        sector_performances_daily = pd.concat([self.long_sectors, self.short_sectors], axis=0)
        sector_performances_daily = sector_performances_daily.reset_index().sort_values(['Date', 'Pred_Rank'], ascending=True).set_index(['Date', 'Sector'], drop=True)
        self.metrics_dfs['セクター別成績（日次）'] = sector_performances_daily.copy()

    def get_monthly_sector_performances(self):
        '''業種ごとの月次リターンの算出'''
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
        self.metrics_dfs['セクター別成績（月次）'] = monthly_sector_performances.copy()

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
        self.metrics_dfs['セクター別成績（トータル）'] = sector_performances_df.sort_values('Total_num', ascending=False)


    '''以下、相関係数関係'''
    def get_spearman_corr(self):
        '''
      日次のsperamanの順位相関係数と，その平均や標準偏差を算出
        '''
        daily_spearman = [spearmanr(x['Target_Rank'], x['Pred_Rank'])[0] for _, x in self.result_df.groupby('Date')]
        dateindex = self.result_df.index.get_level_values('Date').unique()
        self.metrics_dfs['Spearman相関'] = pd.DataFrame(daily_spearman, index=dateindex, columns=['SpearmanCorr'])
        self.metrics_dfs['Spearman相関（集計）'] = self.metrics_dfs['Spearman相関'].describe()

    def get_numerai_corrs(self):
        '''
        日次のnumerai_corrと，その平均や標準偏差を算出
        numerai_corr：上位・下位の予測に重みづけされた，より実践的な指標．
        '''
        #Target順位化の有無（RawReturnとの相関も算出したい．））
        daily_numerai = self.result_df.groupby('Date').apply(lambda x: self._get_daily_numerai_corr(x['Target_Rank'], x['Pred_Rank']))
        daily_numerai_rank = self.result_df.groupby('Date').apply(lambda x: self._get_daily_numerai_rank_corr(x['Target_Rank'], x['Pred_Rank']))
        #データフレーム化
        dateindex = self.result_df.index.get_level_values('Date').unique()
        self.metrics_dfs['Numerai相関'] = pd.concat([daily_numerai, daily_numerai_rank], axis=1)
        self.metrics_dfs['Numerai相関'].columns = ['NumeraiCorr', 'Rank_NumeraiCorr']
        self.metrics_dfs['Numerai相関（集計）'] = self.metrics_dfs['Numerai相関'].describe()

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



class Visualizer:
    """計算結果の可視化を担当"""
    def __init__(self, metrics_calculator: MetricsCalculator):
        self.metrics_dfs = metrics_calculator.metrics_dfs
        self._output_basic_data()

    def _output_basic_data(self):
        #基本データの出力
        print(f'読み込みが完了しました。')
        sharpe_ratio = self.metrics_dfs['日次成績（集計）'].loc["SR", "LS"]
        print('全期間')
        estimated_df = calculate_stats.estimate_population(self.metrics_dfs['日次成績']['LS'], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print('直近1年')
        estimated_1y_df = calculate_stats.estimate_population(self.metrics_dfs['日次成績']['LS'].iloc[-252:], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print('直近3ヶ月')
        estimated_1y_df = calculate_stats.estimate_population(self.metrics_dfs['日次成績']['LS'].iloc[-63:], CI = 0.997, commentout=True)
        print(f'----------------------------------------------------------------------------------')
        print(f'最大ドローダウン (実績)： {round(self.metrics_dfs["日次成績"]["MaxDD"].max() * 100, 3)}%, {self.metrics_dfs["日次成績"]["MaxDDdays"].max()}日')
        print(f'Spearman順位相関： {round(self.metrics_dfs["Spearman相関（集計）"].loc["mean", "SpearmanCorr"], 5)}')
        print(f'Numerai相関： {round(self.metrics_dfs["Numerai相関（集計）"].loc["mean", "NumeraiCorr"], 5)}')
        print(f'----------------------------------------------------------------------------------')

    def display_result(self):
        '''任意のデータフレームを動的に描画'''
        # ウィジェットの構成要素の定義
        dropdown = widgets.Dropdown(
            options=self.metrics_dfs.keys(),
            description='選択：'
        )
        button = widgets.Button(description="表示")
        output = widgets.Output()

        # ボタンクリック時の挙動を定義
        def on_button_click(b):
            selected_df = self.metrics_dfs[dropdown.value]
            with output:
                output.clear_output()  # 以前の出力をクリア
                pd.set_option('display.max_rows', None)  # 表示行数を無制限に設定
                pd.set_option('display.max_columns', None)  # 表示列数を無制限に設定
                display(selected_df)  # 選択されたデータフレームを表示
                pd.reset_option('display.max_rows')  # 表示行数をデフォルトに戻す
                pd.reset_option('display.max_columns')  # 表示列数をデフォルトに戻す

        button.on_click(on_button_click)

        # ウィジェットの表示
        display(widgets.HBox([dropdown, button]), output)  # outputをウィジェットに追加

#%% デバッグ
if __name__ == '__main__':
    ML_DATASET_PATH = f'{Paths.ML_DATASETS_FOLDER}/New48sectors'
    ml_dataset = MLDataset(ML_DATASET_PATH)
    ls_control_handler = LSControlHandler()
    ls_data_handler = LSDataHandler(ml_dataset = ml_dataset, start_day = datetime(2022, 1, 1))
    metrics_calculator = MetricsCalculator(ls_data_handler = ls_data_handler, bin_num = 5)
    visualizer = Visualizer(metrics_calculator)
    visualizer.display_result()