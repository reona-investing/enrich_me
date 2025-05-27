"""
金融データ特有の変換処理実装
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import warnings

from data_processing.core.transformers.base import FinancialTransformer, TimeSeriesTransformer, RankingTransformer


class MomentumTransformer(TimeSeriesTransformer):
    """モメンタム（移動平均リターン）変換器"""
    
    def __init__(self, windows: List[int] = [5, 21], 
                 base_column: str = '1d_return',
                 add_rank: bool = False,
                 exclude_current_day: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.base_column = base_column
        self.add_rank = add_rank
        self.exclude_current_day = exclude_current_day
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for window in self.windows:
            columns.append(f'{window}d_mom')
            if self.add_rank:
                columns.append(f'{window}d_mom_rank')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return [self.base_column]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """モメンタム特徴量を算出"""
        if self.base_column not in df.columns:
            raise ValueError(f"必要なカラム '{self.base_column}' が存在しません")
        
        result = df.copy()
        
        # ウィンドウごとにモメンタムを計算
        for window in self.windows:
            mom_column = f'{window}d_mom'
            
            if self.group_column in df.index.names:
                # セクター別に計算
                if self.exclude_current_day:
                    # 当日を除いたモメンタム計算（ルックアヘッドバイアス防止）
                    momentum = df.groupby(level=self.group_column)[self.base_column].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
                    # 1日シフトして当日の情報を除去
                    result[mom_column] = momentum.groupby(level=self.group_column).shift(1)
                else:
                    # 当日を含むモメンタム計算
                    result[mom_column] = df.groupby(level=self.group_column)[self.base_column].rolling(
                        window=window, min_periods=1
                    ).mean().reset_index(0, drop=True)
            else:
                # 全体で計算
                if self.exclude_current_day:
                    momentum = df[self.base_column].rolling(window=window, min_periods=1).mean()
                    result[mom_column] = momentum.shift(1)
                else:
                    result[mom_column] = df[self.base_column].rolling(window=window, min_periods=1).mean()
            
            # ランキングを追加
            if self.add_rank:
                rank_column = f'{window}d_mom_rank'
                if 'Date' in df.index.names:
                    # 日付別にランキング
                    result[rank_column] = result.groupby(level='Date')[mom_column].rank(
                        ascending=False, method='dense'
                    )
                else:
                    # 全体でランキング
                    result[rank_column] = result[mom_column].rank(ascending=False, method='dense')
        
        return result


class VolatilityTransformer(TimeSeriesTransformer):
    """ボラティリティ（移動標準偏差）変換器"""
    
    def __init__(self, windows: List[int] = [5, 21],
                 base_column: str = '1d_return',
                 add_rank: bool = False,
                 exclude_current_day: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.windows = windows
        self.base_column = base_column
        self.add_rank = add_rank
        self.exclude_current_day = exclude_current_day
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for window in self.windows:
            columns.append(f'{window}d_vola')
            if self.add_rank:
                columns.append(f'{window}d_vola_rank')
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return [self.base_column]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """ボラティリティ特徴量を算出"""
        if self.base_column not in df.columns:
            raise ValueError(f"必要なカラム '{self.base_column}' が存在しません")
        
        result = df.copy()
        
        # ウィンドウごとにボラティリティを計算
        for window in self.windows:
            vola_column = f'{window}d_vola'
            
            if self.group_column in df.index.names:
                # セクター別に計算
                if self.exclude_current_day:
                    # 当日を除いたボラティリティ計算
                    volatility = df.groupby(level=self.group_column)[self.base_column].rolling(
                        window=window, min_periods=2
                    ).std().reset_index(0, drop=True)
                    # 1日シフトして当日の情報を除去
                    result[vola_column] = volatility.groupby(level=self.group_column).shift(1)
                else:
                    # 当日を含むボラティリティ計算
                    result[vola_column] = df.groupby(level=self.group_column)[self.base_column].rolling(
                        window=window, min_periods=2
                    ).std().reset_index(0, drop=True)
            else:
                # 全体で計算
                if self.exclude_current_day:
                    volatility = df[self.base_column].rolling(window=window, min_periods=2).std()
                    result[vola_column] = volatility.shift(1)
                else:
                    result[vola_column] = df[self.base_column].rolling(window=window, min_periods=2).std()
            
            # ランキングを追加
            if self.add_rank:
                rank_column = f'{window}d_vola_rank'
                if 'Date' in df.index.names:
                    # 日付別にランキング（ボラティリティは昇順でランク付け：低ボラ=高ランク）
                    result[rank_column] = result.groupby(level='Date')[vola_column].rank(
                        ascending=True, method='dense'
                    )
                else:
                    # 全体でランキング
                    result[rank_column] = result[vola_column].rank(ascending=True, method='dense')
        
        return result


class CurrencyRelativeStrengthTransformer(FinancialTransformer):
    """通貨相対強度変換器"""
    
    def __init__(self, return_suffix: str = '_1d_return', **kwargs):
        super().__init__(**kwargs)
        self.return_suffix = return_suffix
        self.currency_pairs = [
            'USDJPY', 'EURJPY', 'AUDJPY', 'EURUSD', 'AUDUSD', 'EURAUD'
        ]
        self.target_currencies = ['JPY', 'USD', 'EUR', 'AUD']
    
    @property
    def output_column_names(self) -> List[str]:
        return [f'{currency}{self.return_suffix}' for currency in self.target_currencies]
    
    @property
    def required_input_columns(self) -> List[str]:
        return [f'{pair}{self.return_suffix}' for pair in self.currency_pairs]
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """通貨ペアから相対強度を算出"""
        result = df.copy()
        
        # 必要なカラムの存在確認
        available_pairs = [col.replace(self.return_suffix, '') 
                          for col in df.columns if col.endswith(self.return_suffix)]
        missing_pairs = set(self.currency_pairs) - set(available_pairs)
        
        if missing_pairs:
            warnings.warn(f"一部の通貨ペアが不足しています: {missing_pairs}")
        
        # 相対強度を計算
        try:
            # JPYの相対強度（円高になるほど値が大きくなる）
            jpy_components = []
            if f'USDJPY{self.return_suffix}' in df.columns:
                jpy_components.append(-df[f'USDJPY{self.return_suffix}'])
            if f'EURJPY{self.return_suffix}' in df.columns:
                jpy_components.append(-df[f'EURJPY{self.return_suffix}'])
            if f'AUDJPY{self.return_suffix}' in df.columns:
                jpy_components.append(-df[f'AUDJPY{self.return_suffix}'])
            
            if jpy_components:
                result[f'JPY{self.return_suffix}'] = sum(jpy_components) / len(jpy_components)
            
            # USDの相対強度
            usd_components = []
            if f'USDJPY{self.return_suffix}' in df.columns:
                usd_components.append(df[f'USDJPY{self.return_suffix}'])
            if f'EURUSD{self.return_suffix}' in df.columns:
                usd_components.append(-df[f'EURUSD{self.return_suffix}'])
            if f'AUDUSD{self.return_suffix}' in df.columns:
                usd_components.append(-df[f'AUDUSD{self.return_suffix}'])
            
            if usd_components:
                result[f'USD{self.return_suffix}'] = sum(usd_components) / len(usd_components)
            
            # AUDの相対強度
            aud_components = []
            if f'AUDJPY{self.return_suffix}' in df.columns:
                aud_components.append(df[f'AUDJPY{self.return_suffix}'])
            if f'AUDUSD{self.return_suffix}' in df.columns:
                aud_components.append(df[f'AUDUSD{self.return_suffix}'])
            if f'EURAUD{self.return_suffix}' in df.columns:
                aud_components.append(-df[f'EURAUD{self.return_suffix}'])
            
            if aud_components:
                result[f'AUD{self.return_suffix}'] = sum(aud_components) / len(aud_components)
            
            # EURの相対強度
            eur_components = []
            if f'EURJPY{self.return_suffix}' in df.columns:
                eur_components.append(df[f'EURJPY{self.return_suffix}'])
            if f'EURUSD{self.return_suffix}' in df.columns:
                eur_components.append(df[f'EURUSD{self.return_suffix}'])
            if f'EURAUD{self.return_suffix}' in df.columns:
                eur_components.append(df[f'EURAUD{self.return_suffix}'])
            
            if eur_components:
                result[f'EUR{self.return_suffix}'] = sum(eur_components) / len(eur_components)
            
            # 元の通貨ペアカラムを削除
            for pair in self.currency_pairs:
                pair_column = f'{pair}{self.return_suffix}'
                if pair_column in result.columns:
                    result = result.drop(columns=[pair_column])
        
        except Exception as e:
            warnings.warn(f"通貨相対強度の計算に失敗しました: {e}")
        
        return result


class BondSpreadTransformer(FinancialTransformer):
    """債券スプレッド変換器"""
    
    def __init__(self, return_suffix: str = '_1d_return', **kwargs):
        super().__init__(**kwargs)
        self.return_suffix = return_suffix
        self.bond_pairs = [
            ('USbond10', 'USbond2', 'US_bond_diff'),
            ('JPbond10', 'JPbond2', 'JP_bond_diff'),
            ('GEbond10', 'GEbond2', 'GE_bond_diff')
        ]
    
    @property
    def output_column_names(self) -> List[str]:
        return [f'{spread_name}{self.return_suffix}' for _, _, spread_name in self.bond_pairs]
    
    @property
    def required_input_columns(self) -> List[str]:
        columns = []
        for long_bond, short_bond, _ in self.bond_pairs:
            columns.extend([f'{long_bond}{self.return_suffix}', f'{short_bond}{self.return_suffix}'])
        return columns
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """債券スプレッド（10年債 - 2年債）を算出"""
        result = df.copy()
        
        for long_bond, short_bond, spread_name in self.bond_pairs:
            long_col = f'{long_bond}{self.return_suffix}'
            short_col = f'{short_bond}{self.return_suffix}'
            spread_col = f'{spread_name}{self.return_suffix}'
            
            if long_col in df.columns and short_col in df.columns:
                # スプレッドを計算
                result[spread_col] = df[long_col] - df[short_col]
                
                # 元のカラムを削除
                result = result.drop(columns=[short_col])
            elif long_col in df.columns:
                # 長期債のみ存在する場合はそのまま保持
                pass
            else:
                warnings.warn(f"債券データが不足しています: {long_bond}, {short_bond}")
        
        return result


class RankingTransformer(RankingTransformer):
    """ランキング変換器（金融データ特化版）"""
    
    def __init__(self, target_columns: Optional[List[str]] = None,
                 ranking_method: str = 'dense',
                 ascending: bool = False,
                 suffix: str = '_rank',
                 **kwargs):
        super().__init__(ranking_method=ranking_method, ascending=ascending, **kwargs)
        self.target_columns = target_columns
        self.suffix = suffix
    
    @property
    def output_column_names(self) -> List[str]:
        if self.target_columns:
            return [f'{col}{self.suffix}' for col in self.target_columns]
        else:
            return []  # 実行時に動的に決定
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.target_columns or []
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """各指標の横断面ランキングを算出"""
        result = df.copy()
        
        # 対象カラムを決定
        if self.target_columns:
            target_cols = [col for col in self.target_columns if col in df.columns]
        else:
            # 数値カラムを自動選択（ただし、既にランクカラムは除外）
            target_cols = [
                col for col in df.select_dtypes(include=[np.number]).columns 
                if not col.endswith('_rank') and not col.endswith('_cat')
            ]
        
        # 各カラムについてランキングを計算
        for col in target_cols:
            rank_col = f'{col}{self.suffix}'
            
            if 'Date' in df.index.names:
                # 日付別にランキング
                result[rank_col] = df.groupby(level='Date')[col].rank(
                    method=self.ranking_method, ascending=self.ascending
                )
            else:
                # 全体でランキング
                result[rank_col] = df[col].rank(
                    method=self.ranking_method, ascending=self.ascending
                )
        
        return result


class SectorCategoricalTransformer(FinancialTransformer):
    """セクターカテゴリ変数変換器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sector_mapping = {}
    
    @property
    def output_column_names(self) -> List[str]:
        return ['Sector_cat']
    
    @property
    def required_input_columns(self) -> List[str]:
        return []  # インデックスのSectorを使用
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """セクターを数値カテゴリ変数に変換"""
        result = df.copy()
        
        if 'Sector' in df.index.names:
            # セクター名を取得
            sectors = df.index.get_level_values('Sector').unique()
            
            # セクターを数値にマッピング
            self.sector_mapping = {sector: i for i, sector in enumerate(sorted(sectors))}
            
            # カテゴリ変数として追加
            result['Sector_cat'] = df.index.get_level_values('Sector').map(self.sector_mapping)
        else:
            warnings.warn("Sectorインデックスが見つかりません")
        
        return result


class TechnicalIndicatorTransformer(TimeSeriesTransformer):
    """テクニカル指標変換器"""
    
    def __init__(self, indicators: List[str] = ['rsi', 'macd', 'bb'],
                 price_columns: List[str] = ['Close'],
                 **kwargs):
        super().__init__(**kwargs)
        self.indicators = indicators
        self.price_columns = price_columns
    
    @property
    def output_column_names(self) -> List[str]:
        columns = []
        for indicator in self.indicators:
            if indicator == 'rsi':
                columns.append('RSI_14')
            elif indicator == 'macd':
                columns.extend(['MACD', 'MACD_Signal', 'MACD_Histogram'])
            elif indicator == 'bb':
                columns.extend(['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width'])
        return columns
    
    @property
    def required_input_columns(self) -> List[str]:
        return self.price_columns
    
    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """各種テクニカル指標を算出"""
        result = df.copy()
        
        for price_col in self.price_columns:
            if price_col not in df.columns:
                continue
            
            # RSI
            if 'rsi' in self.indicators:
                result = self._calculate_rsi(result, price_col)
            
            # MACD
            if 'macd' in self.indicators:
                result = self._calculate_macd(result, price_col)
            
            # ボリンジャーバンド
            if 'bb' in self.indicators:
                result = self._calculate_bollinger_bands(result, price_col)
        
        return result
    
    def _calculate_rsi(self, df: pd.DataFrame, price_col: str, period: int = 14) -> pd.DataFrame:
        """RSIを計算"""
        result = df.copy()
        
        if self.group_column in df.index.names:
            # セクター別に計算
            def calc_rsi_group(group):
                delta = group[price_col].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))
            
            result['RSI_14'] = df.groupby(level=self.group_column).apply(calc_rsi_group).reset_index(0, drop=True)
        else:
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            result['RSI_14'] = 100 - (100 / (1 + rs))
        
        return result
    
    def _calculate_macd(self, df: pd.DataFrame, price_col: str) -> pd.DataFrame:
        """MACDを計算"""
        result = df.copy()
        
        if self.group_column in df.index.names:
            # セクター別に計算
            def calc_macd_group(group):
                ema_12 = group[price_col].ewm(span=12).mean()
                ema_26 = group[price_col].ewm(span=26).mean()
                macd_line = ema_12 - ema_26
                signal_line = macd_line.ewm(span=9).mean()
                histogram = macd_line - signal_line
                return pd.DataFrame({
                    'MACD': macd_line,
                    'MACD_Signal': signal_line,
                    'MACD_Histogram': histogram
                })
            
            macd_result = df.groupby(level=self.group_column).apply(calc_macd_group)
            result[['MACD', 'MACD_Signal', 'MACD_Histogram']] = macd_result.reset_index(0, drop=True)
        else:
            ema_12 = df[price_col].ewm(span=12).mean()
            ema_26 = df[price_col].ewm(span=26).mean()
            result['MACD'] = ema_12 - ema_26
            result['MACD_Signal'] = result['MACD'].ewm(span=9).mean()
            result['MACD_Histogram'] = result['MACD'] - result['MACD_Signal']
        
        return result
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, price_col: str, 
                                 period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """ボリンジャーバンドを計算"""
        result = df.copy()
        
        if self.group_column in df.index.names:
            # セクター別に計算
            def calc_bb_group(group):
                middle = group[price_col].rolling(window=period).mean()
                std = group[price_col].rolling(window=period).std()
                upper = middle + (std * std_dev)
                lower = middle - (std * std_dev)
                width = upper - lower
                return pd.DataFrame({
                    'BB_Upper': upper,
                    'BB_Middle': middle,
                    'BB_Lower': lower,
                    'BB_Width': width
                })
            
            bb_result = df.groupby(level=self.group_column).apply(calc_bb_group)
            result[['BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Width']] = bb_result.reset_index(0, drop=True)
        else:
            middle = df[price_col].rolling(window=period).mean()
            std = df[price_col].rolling(window=period).std()
            result['BB_Upper'] = middle + (std * std_dev)
            result['BB_Middle'] = middle
            result['BB_Lower'] = middle - (std * std_dev)
            result['BB_Width'] = result['BB_Upper'] - result['BB_Lower']
        
        return result


# ファクトリー関数
def get_financial_transformer(transformer_type: str, **kwargs) -> FinancialTransformer:
    """金融変換器タイプに応じた変換器を取得"""
    transformers = {
        'momentum': MomentumTransformer,
        'volatility': VolatilityTransformer,
        'currency_relative_strength': CurrencyRelativeStrengthTransformer,
        'bond_spread': BondSpreadTransformer,
        'ranking': RankingTransformer,
        'sector_categorical': SectorCategoricalTransformer,
        'technical_indicators': TechnicalIndicatorTransformer
    }
    
    if transformer_type not in transformers:
        raise ValueError(f"未対応の金融変換器タイプです: {transformer_type}")
    
    transformer_class = transformers[transformer_type]
    return transformer_class(**kwargs)


# 便利な組み合わせ関数
def create_standard_financial_pipeline_transformers(
    momentum_windows: List[int] = [5, 21],
    volatility_windows: List[int] = [5, 21],
    add_ranking: bool = True,
    add_sector_categorical: bool = True
) -> List[FinancialTransformer]:
    """標準的な金融特徴量変換器のセットを作成"""
    
    transformers = []
    
    # モメンタム変換器
    transformers.append(MomentumTransformer(
        windows=momentum_windows,
        add_rank=add_ranking
    ))
    
    # ボラティリティ変換器
    transformers.append(VolatilityTransformer(
        windows=volatility_windows,
        add_rank=add_ranking
    ))
    
    # ランキング変換器（全特徴量対象）
    if add_ranking:
        transformers.append(RankingTransformer())
    
    # セクターカテゴリ変換器
    if add_sector_categorical:
        transformers.append(SectorCategoricalTransformer())
    
    return transformers