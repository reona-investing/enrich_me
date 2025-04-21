import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.stats import spearmanr, norm


def calculate_metrics(pred_result_df: pd.DataFrame, metrics: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    予測結果の評価指標を計算する
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        metrics: 計算する評価指標のリスト（省略時は全指標）
        
    Returns:
        評価指標の辞書
    """
    available_metrics = {
        'spearman': calculate_spearman_correlation,
        'numerai': calculate_numerai_correlation,
        'rmse': calculate_rmse,
        'mae': calculate_mae,
        'quantile_return': calculate_return_by_quantile,
        'longshort': calculate_longshort_performance
    }
    
    if metrics is None:
        metrics = list(available_metrics.keys())
    
    result = {}
    for metric in metrics:
        if metric in available_metrics:
            result[metric] = available_metrics[metric](pred_result_df)
        else:
            print(f"警告: 未知の評価指標 '{metric}' はスキップされました。")
    
    return result


def calculate_spearman_correlation(pred_result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    日次のSpearman順位相関係数と、その平均や標準偏差を算出
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        
    Returns:
        Spearman相関の評価結果の辞書
    """
    daily_correlations = [
        spearmanr(x['Target'], x['Pred'])[0] 
        for _, x in pred_result_df.groupby('Date')
    ]
    
    date_index = pred_result_df.index.get_level_values('Date').unique()
    corr_df = pd.DataFrame(daily_correlations, index=date_index, columns=['SpearmanCorr'])
    
    return {
        'daily': corr_df,
        'mean': np.mean(daily_correlations),
        'std': np.std(daily_correlations),
        'max': np.max(daily_correlations),
        'min': np.min(daily_correlations),
        'summary': corr_df.describe()
    }


def calculate_numerai_correlation(pred_result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    日次のNumerai相関と、その平均や標準偏差を算出
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        
    Returns:
        Numerai相関の評価結果の辞書
    """
    # 通常のNumerai相関
    daily_numerai = pred_result_df.groupby('Date').apply(
        lambda x: _calc_daily_numerai_corr(x['Target'], x['Pred'])
    )
    
    # 目的変数をランク化したNumerai相関
    daily_numerai_rank = pred_result_df.groupby('Date').apply(
        lambda x: _calc_daily_numerai_rank_corr(
            x['Target'].rank(), x['Pred'].rank()
        )
    )
    
    # データフレーム化
    date_index = pred_result_df.index.get_level_values('Date').unique()
    numerai_df = pd.DataFrame({
        'NumeraiCorr': daily_numerai,
        'Rank_NumeraiCorr': daily_numerai_rank
    }, index=date_index)
    
    return {
        'daily': numerai_df,
        'mean': daily_numerai.mean(),
        'std': daily_numerai.std(),
        'mean_rank': daily_numerai_rank.mean(),
        'std_rank': daily_numerai_rank.std(),
        'summary': numerai_df.describe()
    }


def _calc_daily_numerai_corr(target: pd.Series, pred: pd.Series) -> float:
    """
    日次のNumerai相関を算出
    
    Args:
        target: 目的変数
        pred: 予測値
        
    Returns:
        Numerai相関値
    """
    # 予測値の変換
    pred_array = np.array(pred)
    scaled_pred = (pred_array - pred_array.min()) / (pred_array.max() - pred_array.min() + 1e-8)
    scaled_pred = scaled_pred * 0.98 + 0.01  # [0.01, 0.99]の範囲に収める
    gauss_pred = norm.ppf(scaled_pred)
    pred_p15 = np.sign(gauss_pred) * np.abs(gauss_pred) ** 1.5
    
    # 目的変数の変換
    target_array = np.array(target)
    centered_target = target_array - target_array.mean()
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    
    return np.corrcoef(pred_p15, target_p15)[0, 1]


def _calc_daily_numerai_rank_corr(target_rank: pd.Series, pred_rank: pd.Series) -> float:
    """
    日次のNumerai相関（ランク化版）を算出
    
    Args:
        target_rank: 目的変数のランク
        pred_rank: 予測値のランク
        
    Returns:
        Numerai相関値（ランク化版）
    """
    # 目的変数と予測値のそれぞれをランクベースで変換
    processed_ranks = []
    for x in [target_rank, pred_rank]:
        x_array = np.array(x)
        scaled_x = (x_array - 0.5) / len(x_array)
        gauss_x = norm.ppf(scaled_x)
        x_p15 = np.sign(gauss_x) * np.abs(gauss_x) ** 1.5
        processed_ranks.append(x_p15)
    
    return np.corrcoef(processed_ranks[0], processed_ranks[1])[0, 1]


def calculate_rmse(pred_result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    平均二乗誤差の平方根（RMSE）を計算
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        
    Returns:
        RMSE評価結果の辞書
    """
    daily_rmse = pred_result_df.groupby('Date').apply(
        lambda x: np.sqrt(np.mean((x['Target'] - x['Pred']) ** 2))
    )
    
    return {
        'daily': daily_rmse,
        'mean': daily_rmse.mean(),
        'std': daily_rmse.std(),
        'total': np.sqrt(np.mean((pred_result_df['Target'] - pred_result_df['Pred']) ** 2))
    }


def calculate_mae(pred_result_df: pd.DataFrame) -> Dict[str, Any]:
    """
    平均絶対誤差（MAE）を計算
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        
    Returns:
        MAE評価結果の辞書
    """
    daily_mae = pred_result_df.groupby('Date').apply(
        lambda x: np.mean(np.abs(x['Target'] - x['Pred']))
    )
    
    return {
        'daily': daily_mae,
        'mean': daily_mae.mean(),
        'std': daily_mae.std(),
        'total': np.mean(np.abs(pred_result_df['Target'] - pred_result_df['Pred']))
    }


def calculate_return_by_quantile(pred_result_df: pd.DataFrame, n_quantiles: int = 5) -> Dict[str, Any]:
    """
    予測値の分位数ごとの目的変数の平均リターンを計算
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        n_quantiles: 分位数の数
        
    Returns:
        分位数ごとのリターン評価結果の辞書
    """
    df = pred_result_df.copy()
    
    # 各日付ごとに予測値の分位数を計算
    df['Pred_Quantile'] = df.groupby('Date')['Pred'].transform(
        lambda x: pd.qcut(x.rank(method='first'), q=n_quantiles, labels=False)
    )
    
    # 分位数ごとのリターンを計算
    quantile_returns = df.groupby(['Date', 'Pred_Quantile'])['Target'].mean().unstack()
    quantile_summary = quantile_returns.mean()
    
    return {
        'daily': quantile_returns,
        'mean': quantile_summary,
        'spread': quantile_summary.iloc[-1] - quantile_summary.iloc[0]
    }


def calculate_longshort_performance(pred_result_df: pd.DataFrame, n_sectors: int = 3, top_slope: float = 1.0) -> Dict[str, Any]:
    """
    ロング・ショート戦略のパフォーマンスを計算
    
    Args:
        pred_result_df: 予測結果のデータフレーム（Target列とPred列を含む）
        n_sectors: 上位・下位何業種を取引対象とするか
        top_slope: 最上位（最下位）予想業種にどれだけの比率で傾斜を掛けるか
        
    Returns:
        ロング・ショート戦略のパフォーマンス評価結果の辞書
    """
    df = pred_result_df.copy()
    
    # 各日付ごとに予測値のランクを計算
    df['Pred_Rank'] = df.groupby('Date')['Pred'].rank()
    
    # ショートポジション（下位n業種）
    short_positions = -df[df['Pred_Rank'] <= n_sectors].copy()
    # 最下位業種に傾斜をかける
    short_positions.loc[short_positions['Pred_Rank'] == short_positions['Pred_Rank'].min(), 'Target'] *= top_slope
    if n_sectors > 1:  # 複数業種がある場合は他の業種の重みを調整
        short_positions.loc[short_positions['Pred_Rank'] != short_positions['Pred_Rank'].min(), 'Target'] *= (1 - (top_slope - 1) / (n_sectors - 1))
    # 日次の平均リターンを計算
    short_return = short_positions.groupby('Date')['Target'].mean()
    
    # ロングポジション（上位n業種）
    pred_max_rank = df['Pred_Rank'].max()
    long_positions = df[df['Pred_Rank'] > pred_max_rank - n_sectors].copy()
    # 最上位業種に傾斜をかける
    long_positions.loc[long_positions['Pred_Rank'] == long_positions['Pred_Rank'].max(), 'Target'] *= top_slope
    if n_sectors > 1:  # 複数業種がある場合は他の業種の重みを調整
        long_positions.loc[long_positions['Pred_Rank'] != long_positions['Pred_Rank'].max(), 'Target'] *= (1 - (top_slope - 1) / (n_sectors - 1))
    # 日次の平均リターンを計算
    long_return = long_positions.groupby('Date')['Target'].mean()
    
    # ロング・ショート合成リターン
    performance_df = pd.DataFrame({
        'Long': long_return,
        'Short': short_return
    })
    performance_df['LS'] = (performance_df['Long'] + performance_df['Short']) / 2
    
    # パフォーマンスサマリー
    performance_summary = performance_df.describe()
    performance_summary.loc['SR'] = performance_summary.loc['mean'] / performance_summary.loc['std']
    
    # 累積リターン
    performance_df['L_Cumprod'] = (performance_df['Long'] + 1).cumprod() - 1
    performance_df['S_Cumprod'] = (performance_df['Short'] + 1).cumprod() - 1
    performance_df['LS_Cumprod'] = (performance_df['LS'] + 1).cumprod() - 1
    
    # ドローダウン計算
    performance_df['DD'] = 1 - (performance_df['LS_Cumprod'] + 1) / (performance_df['LS_Cumprod'].cummax() + 1)
    performance_df['MaxDD'] = performance_df['DD'].cummax()
    performance_df['DDdays'] = performance_df.groupby((performance_df['DD'] == 0).cumsum()).cumcount()
    performance_df['MaxDDdays'] = performance_df['DDdays'].cummax()
    
    return {
        'daily': performance_df,
        'summary': performance_summary,
        'final_return': performance_df['LS_Cumprod'].iloc[-1],
        'sharpe_ratio': performance_summary.loc['SR', 'LS'],
        'max_drawdown': performance_df['MaxDD'].max(),
        'max_drawdown_days': performance_df['MaxDDdays'].max()
    }