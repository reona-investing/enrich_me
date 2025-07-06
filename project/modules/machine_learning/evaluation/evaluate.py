import pandas as pd
from typing import Literal
from datetime import datetime
import numpy as np
from scipy.stats import spearmanr, norm
from machine_learning.evaluation import calculate_stats
import ipywidgets as widgets
from IPython.display import display

from utils.paths import Paths
from utils.timeseries import Duration


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
                 pred_result_df: pd.DataFrame,
                 raw_target_df: pd.DataFrame,
                 start_day: datetime = datetime(2013, 1, 1),
                 end_day: datetime = datetime.today(),
                 ls_control: LSControlHandler = LSControlHandler(),
                 ):
        self.duration = Duration(start=start_day, end=end_day)
        pred_result_df = self.duration.extract_from_df(pred_result_df, 'Date')
        raw_target_df = self.duration.extract_from_df(raw_target_df, 'Date')
        self.control_df = self.duration.extract_from_df(ls_control.control_df, 'Date')
        self.control_type = ls_control.control_type
        self.original_column_names = pred_result_df.columns
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
                sectors_to_trade_count: int = 3,
                quantile_bin_count: int = None,
                top_sector_weight: float = 1):
        '''
        Long-Short戦略のモデルの予測結果を格納し、各種指標を計算します。
        Args:
            ls_data_handler (LSDataHandler): ロングショート戦略の各種データを保持するクラス
            sectors_to_trade_count (int): 上位（下位）何sectorを取引の対象とするか。
            quantile_bin_count (int):分位リターンのbin数。指定しない場合はsector数が割り当てられる。
            top_sector_weight (float): 最上位（最下位）予想業種にどれだけの比率で傾斜を掛けるか？
        '''
        self.sectors_to_trade_count = sectors_to_trade_count
        self.result_df = ls_data_handler.result_df
        self.control_df = ls_data_handler.control_df
        self.control_type = ls_data_handler.control_type

        if quantile_bin_count is None:
            quantile_bin_count = int(self.result_df[['RawTarget_Rank']].max().iloc[0] + 1)
        self.quantile_bin_count = quantile_bin_count
        self.top_sector_weight = top_sector_weight
        self.original_column_names = ls_data_handler.original_column_names
        self.metrics_dfs = {}  # 辞書キー名は日本語のままに！
        
        self._calculate_all_metrics()
    
    def _calculate_all_metrics(self):
        self.calculate_return_by_quantile_bin()
        self.calculate_longshort_performance()
        self.calculate_longshort_probability()
        self.calculate_longshort_minus_control()
        self.calculate_monthly_longshort()
        self.calculate_daily_sector_performances()
        self.calculate_monthly_sector_performances()
        self.calculate_total_sector_performances()
        self.calculate_spearman_correlation()
        self.calculate_numerai_correlationss()
    

    def calculate_return_by_quantile_bin(self) -> pd.DataFrame:
        '''指定したbin数ごとのリターンを算出。'''
        df = self.result_df.dropna().copy()
        for column in self.original_column_names:
            df[f'{column}_Bin'] = \
              df.groupby('Date')[f'{column}_Rank'].transform(lambda x: pd.qcut(x, q=self.quantile_bin_count, labels=False))
        self.metrics_dfs['分位成績'] = df.groupby(['Date', 'Pred_Bin'])[['RawTarget']].mean().unstack(-1)
        self.metrics_dfs['分位成績（集計）'] = self.metrics_dfs['分位成績'].stack(future_stack=True).groupby('Pred_Bin').describe().T


    def calculate_longshort_performance(self) -> pd.DataFrame:
        '''ロング・ショートそれぞれの結果を算出する。'''
        df = self.result_df.copy()
        short_positions = - df.loc[df['Pred_Rank'] <= self.sectors_to_trade_count]
        if self.sectors_to_trade_count > 1:
            short_positions.loc[short_positions['Pred_Rank'] == short_positions['Pred_Rank'].min(), 'RawTarget'] *= self.top_sector_weight
            short_positions.loc[short_positions['Pred_Rank'] != short_positions['Pred_Rank'].min(), 'RawTarget'] *= (1 - (self.top_sector_weight - 1) / (self.sectors_to_trade_count - 1))
        short_positions_return = short_positions.groupby('Date')[['RawTarget']].mean()
        short_positions_return = short_positions_return.rename(columns={'RawTarget':'Short'})
        long_positions = df.loc[df['Pred_Rank'] > max(df['Pred_Rank']) - self.sectors_to_trade_count]
        if self.sectors_to_trade_count > 1:
            long_positions.loc[long_positions['Pred_Rank'] == long_positions['Pred_Rank'].max(), 'RawTarget'] *= self.top_sector_weight
            long_positions.loc[long_positions['Pred_Rank'] != long_positions['Pred_Rank'].max(), 'RawTarget'] *= (1 - (self.top_sector_weight - 1) / (self.sectors_to_trade_count - 1))
        long_positions_return = long_positions.groupby('Date')[['RawTarget']].mean()
        long_positions_return = long_positions_return.rename(columns={'RawTarget':'Long'})
        performance_df = pd.concat([long_positions_return, short_positions_return], axis=1)
        performance_df['LS'] = (performance_df['Long'] + performance_df['Short']) / 2

        performance_summary = performance_df.describe()
        performance_summary.loc['SR'] = \
          performance_summary.loc['mean'] / performance_summary.loc['std']

        performance_df['L_Cumprod'] = (performance_df['Long'] + 1).cumprod() - 1
        performance_df['S_Cumprod'] = (performance_df['Short'] + 1).cumprod() - 1
        performance_df['LS_Cumprod'] = (performance_df['LS'] + 1).cumprod() - 1
        performance_df['DD'] = 1 - (performance_df['LS_Cumprod'] + 1) / (performance_df['LS_Cumprod'].cummax() + 1)
        performance_df['MaxDD'] = performance_df['DD'].cummax()
        performance_df['DDdays'] = performance_df.groupby((performance_df['DD'] == 0).cumsum()).cumcount()
        performance_df['MaxDDdays'] = performance_df['DDdays'].cummax()
        performance_df = performance_df[['LS', 'LS_Cumprod',
                                     'DD', 'MaxDD', 'DDdays', 'MaxDDdays',
                                     'Long', 'Short',	'L_Cumprod', 'S_Cumprod']]
        self.metrics_dfs['日次成績'] = performance_df
        self.metrics_dfs['日次成績（集計）'] = performance_summary


    def calculate_longshort_probability(self) -> pd.DataFrame:
        '''Long, Short, LSの各リターンを算出する。'''
        performance_df = self.metrics_dfs['日次成績'][['LS']].copy()
        performance_df['CumMean'] = performance_df['LS'].expanding().mean().shift(1)
        performance_df['CumStd'] = performance_df['LS'].expanding().std().shift(1)
        performance_df['DegFreedom'] = (performance_df['LS'].expanding().count() - 1).shift(1)
        performance_df = performance_df.dropna(axis=0)
        # 前日までのリターンの結果で、その時点での母平均と母標準偏差の信頼区間を算出
        performance_df['MeanTuple'] = \
          performance_df.apply(lambda row: calculate_stats.estimate_population_mean(row['CumMean'], row['CumStd'], CI=0.997, deg_freedom=row['DegFreedom']), axis=1)
        performance_df['StdTuple'] = \
          performance_df.apply(lambda row: calculate_stats.estimate_population_std(row['CumStd'], CI=0.997, deg_freedom=row['DegFreedom']), axis=1)
        performance_df[['MeanWorst', 'MeanBest']] = performance_df['MeanTuple'].apply(pd.Series)
        performance_df[['StdWorst', 'StdBest']] = performance_df['StdTuple'].apply(pd.Series)
        # 当日のリターンがどの程度の割合で起こるのかを算出
        performance_df['Probability'] = round(performance_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['CumMean'], scale=row['CumStd'])), axis=1), 6)
        performance_df['ProbabWorst'] = round(performance_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['MeanWorst'], scale=row['StdWorst'])), axis=1), 6)
        performance_df['ProbabBest'] = round(performance_df.apply(lambda row: float(norm.cdf(row['LS'], loc=row['MeanBest'], scale=row['StdBest'])), axis=1), 6)
        performance_df = performance_df[['LS', 'Probability', 'ProbabWorst', 'ProbabBest']]
        self.metrics_dfs['日次成績（確率分布）'] = performance_df


    def calculate_longshort_minus_control(self):
        '''コントロールを差し引いたリターン結果を表示する。'''
        performance_df = pd.merge(self.metrics_dfs['日次成績'], self.control_df[[f'{self.control_type}_daytime']], how='left', left_index=True, right_index=True)
        performance_df['Long'] -= performance_df[f'{self.control_type}_daytime']
        performance_df['Short'] += performance_df[f'{self.control_type}_daytime']

        performance_summary = performance_df[['Long', 'Short', 'LS']].describe()
        performance_summary.loc['SR'] = \
          performance_summary.loc['mean'] / performance_summary.loc['std']

        performance_df['L_Cumprod'] = (performance_df['Long'] + 1).cumprod() - 1
        performance_df['S_Cumprod'] = (performance_df['Short'] + 1).cumprod() - 1
        performance_df = performance_df[['LS', 'LS_Cumprod',
                                     'DD', 'MaxDD', 'DDdays', 'MaxDDdays',
                                     'Long', 'Short',	'L_Cumprod', 'S_Cumprod']]
        self.metrics_dfs[f'日次成績（{self.control_type}差分）'] = performance_df
        self.metrics_dfs[f'日次成績（集計・{self.control_type}差分）'] = performance_summary


    def calculate_monthly_longshort(self):
        '''月次のリターンを算出'''
        monthly_longshort = (self.metrics_dfs['日次成績'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort_sub_control = (self.metrics_dfs[f'日次成績（{self.control_type}差分）'][['Long', 'Short', 'LS']] + 1).resample('ME').prod() - 1
        monthly_longshort['Cumprod'] = (monthly_longshort['LS'] + 1).cumprod() - 1
        monthly_longshort_sub_control['Cumprod'] = (monthly_longshort_sub_control['LS'] + 1).cumprod() - 1
        self.metrics_dfs['月次成績'] = monthly_longshort
        self.metrics_dfs[f'月次成績（{self.control_type}差分）'] = monthly_longshort_sub_control


    def calculate_daily_sector_performances(self):
        '''業種ごとの成績を算出する'''
        df = self.result_df.copy()
        long_theshold = len(df.index.get_level_values('Sector').unique()) - self.sectors_to_trade_count + 1
        self.long_sectors = df[df['Pred_Rank'] >= long_theshold]
        self.short_sectors = df[df['Pred_Rank'] <= self.sectors_to_trade_count]
        #日次リターンの算出
        sector_performances_daily = pd.concat([self.long_sectors, self.short_sectors], axis=0)
        sector_performances_daily = sector_performances_daily.reset_index().sort_values(['Date', 'Pred_Rank'], ascending=True).set_index(['Date', 'Sector'], drop=True)
        self.metrics_dfs['セクター別成績（日次）'] = sector_performances_daily.copy()

    def calculate_monthly_sector_performances(self):
        '''業種ごとの月次リターンの算出'''
        long_sectors = self.long_sectors.copy()
        short_sectors = self.short_sectors.copy()
        long_sectors = long_sectors.loc[long_sectors.index.get_level_values('Date') != long_sectors.index.get_level_values('Date')[-1]]
        short_sectors = short_sectors.loc[short_sectors.index.get_level_values('Date') != short_sectors.index.get_level_values('Date')[-1]]
        long_sectors['RawTarget'] += 1
        short_sectors['RawTarget'] -= 1
        short_sectors['RawTarget'] = short_sectors['RawTarget'].abs()
        long_grouped  = long_sectors.groupby([pd.Grouper(level='Date', freq='ME'), 'Sector'])
        short_grouped = short_sectors.groupby([pd.Grouper(level='Date', freq='ME'), 'Sector'])
        # ①月ごとのデータの件数
        long_count = pd.DataFrame(long_grouped.size(), columns=['Long_Num'])
        short_count = pd.DataFrame(short_grouped.size(), columns=['Short_Num'])
        # ②月ごとの'Target'列のすべての値の積
        long_return = long_grouped[['RawTarget']].apply(np.prod)
        short_return = short_grouped[['RawTarget']].apply(np.prod)

        long_monthly = pd.concat([long_count, long_return], axis=1)
        short_monthly = pd.concat([short_count, short_return], axis=1)
        monthly_sector_performances = pd.merge(long_monthly, short_monthly, how='outer', left_index=True, right_index=True).fillna(0)
        self.metrics_dfs['セクター別成績（月次）'] = monthly_sector_performances.copy()

    def calculate_total_sector_performances(self):
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


    def calculate_spearman_correlation(self):
        """
        日次のSpearman順位相関係数と、その平均や標準偏差を算出
        """
        daily_correlations = [spearmanr(x['Target_Rank'], x['Pred_Rank'])[0] for _, x in self.result_df.groupby('Date')]
        date_index = self.result_df.index.get_level_values('Date').unique()
        self.metrics_dfs['Spearman相関'] = pd.DataFrame(daily_correlations, index=date_index, columns=['SpearmanCorr'])
        self.metrics_dfs['Spearman相関（集計）'] = self.metrics_dfs['Spearman相関'].describe()

    def calculate_numerai_correlationss(self):
        '''
        日次のnumerai_corrと，その平均や標準偏差を算出
        numerai_corr：上位・下位の予測に重みづけされた，より実践的な指標．
        '''
        #Target順位化の有無（RawReturnとの相関も算出したい．））
        daily_numerai = self.result_df.groupby('Date').apply(lambda x: self._calc_daily_numerai_corr(x['Target_Rank'], x['Pred_Rank']))
        daily_numerai_rank = self.result_df.groupby('Date').apply(lambda x: self._calc_daily_numerai_rank_corr(x['Target_Rank'], x['Pred_Rank']))
        #データフレーム化
        dateindex = self.result_df.index.get_level_values('Date').unique()
        self.metrics_dfs['Numerai相関'] = pd.concat([daily_numerai, daily_numerai_rank], axis=1)
        self.metrics_dfs['Numerai相関'].columns = ['NumeraiCorr', 'Rank_NumeraiCorr']
        self.metrics_dfs['Numerai相関（集計）'] = self.metrics_dfs['Numerai相関'].describe()

    def _calc_daily_numerai_corr(self, target: pd.Series, pred_rank: pd.Series):
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

    def _calc_daily_numerai_rank_corr(self, target_rank: pd.Series, pred_rank: pd.Series):
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
    from machine_learning.ml_dataset import MLDataset
    ML_DATASET_PATH = f'{Paths.ML_DATASETS_FOLDER}/48sectors_Ensembled_learned_in_250308'
    ml_dataset = MLDataset(ML_DATASET_PATH)
    ls_control_handler = LSControlHandler()
    ls_data_handler = LSDataHandler(ml_dataset.evaluation_materials.pred_result_df, 
                                    ml_dataset.evaluation_materials.raw_target_df, 
                                    start_day = datetime(2022, 1, 1))
    metrics_calculator = MetricsCalculator(ls_data_handler = ls_data_handler, quantile_bin_count = 5)
    visualizer = Visualizer(metrics_calculator)
    visualizer.display_result()