#%% モジュールのインポート
import pandas as pd
from scipy.stats import t, chi2

#%% 関数群
def estimate_population_mean(mean: float, std: float, CI: float, deg_freedom: int):
    '''
    指定した信頼区間での母平均のベストケースとワーストケースを推定。
    mean: 標本平均
    std: 標本標準偏差
    CI：信頼区間（両側）
    deg_freedom: 自由度
    '''
    lower = (1 - CI) / 2
    upper = 1 - lower
    mean_worst = mean + t.ppf(lower, deg_freedom) * (std ** 2 / (deg_freedom + 1)) ** 0.5
    mean_best = mean + t.ppf(upper, deg_freedom) * (std ** 2 / (deg_freedom + 1)) ** 0.5
    return mean_worst, mean_best


def estimate_population_std(std: float, CI: float, deg_freedom: int):
    '''
    指定した信頼区間での母標準偏差のベストケースとワーストケースを推定。
    mean: 標本平均
    std: 標本標準偏差
    CI：信頼区間（両側）
    deg_freedom: 自由度
    '''
    lower = (1 - CI) / 2
    upper = 1 - lower
    std_worst = (deg_freedom * std ** 2 / chi2.ppf(lower, deg_freedom)) ** 0.5
    std_best = (deg_freedom * std ** 2 / chi2.ppf(upper, deg_freedom)) ** 0.5
    return std_worst, std_best

def calculate_DD(mean: float, std: float):
    maxdd = 9 / 4 * (std ** 2) / mean
    maxdd_days = round(9 * (std ** 2) / (mean ** 2))
    return maxdd, maxdd_days

def estimate_population(series: pd.Series, CI: float = 0.997, commentout: bool=False):
    '''
    標本から母平均と母標準偏差を区間推定する。
    series：標本
    CI：信頼区間
    '''
    # 基本的な統計量を算出する。
    mean = series.mean()
    std = series.std()
    SR = mean / std
    sample_size = len(series)
    deg_freedom = sample_size - 1

    # 母平均と母標準偏差, sharpe ratioのベストケースとワーストケースを算出
    mean_worst, mean_best = estimate_population_mean(mean, std, CI, deg_freedom)
    std_worst, std_best = estimate_population_std(std, CI, deg_freedom)

    SR_worst = mean_worst / std_worst
    SR_best = mean_best / std_best

    #理論上のドローダウンを算出
    maxdd, maxdd_days = calculate_DD(mean, std)
    maxdd_worst, maxdd_days_worst = calculate_DD(mean_worst, std_worst)
    maxdd_best, maxdd_days_best = calculate_DD(mean_best, std_best)

    #ベストケースとワーストケースをまとめてdf化
    estimated_dict = {
        'mean': [mean, mean_worst, mean_best],
        'std': [std, std_worst, std_best],
        'SR': [SR, SR_worst, SR_best],
        'MaxDD': [maxdd, maxdd_worst, maxdd_best],
        'MaxDDdays': [maxdd_days, maxdd_days_worst, maxdd_days_best]}
    estimated_df = pd.DataFrame(estimated_dict, columns=['Sample', 'Worst', 'Best'])

    #一応コメントアウト
    if commentout:
        print(f'母集団の推定（{CI * 100}%）')
        print(f'平均リターン: {round(mean * 100, 4)}% (worst: {round(mean_worst * 100, 4)}%, best: {round(mean_best * 100, 4)}%)')
        print(f'標準偏差: {round(std, 6)} (worst:{round(std_worst, 6)}, best:{round(std_best, 6)})')
        print('-----------------------------------------')
        print(f'モデルのスペック')
        print(f'シャープレシオ: {round(SR, 6)} (worst:{round(SR_worst, 6)}, best:{round(SR_best, 6)})')
        print(f'最大ドローダウン: {round(maxdd * 100, 4)}% (worst:{round(maxdd_worst * 100, 4)}%, best:{round(maxdd_best * 100, 4)}%)')
        print(f'最大ドローダウン日数: {round(maxdd_days, 0)}日 (worst:{round(maxdd_days_worst, 0)}日, best:{round(maxdd_days_best, 0)}日)')

    return estimated_df

