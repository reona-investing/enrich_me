"""
データ生成器の実装
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
import warnings

from .base import DataGenerator


class SectorIndexGenerator(DataGenerator):
    """セクターインデックス生成器"""
    
    def __init__(self, sector_calculator=None, **kwargs):
        super().__init__()
        # 既存のSectorIndexCalculatorを使用
        if sector_calculator is None:
            from calculation.sector_index_calculator import SectorIndexCalculator
            self.sector_calculator = SectorIndexCalculator()
        else:
            self.sector_calculator = sector_calculator
    
    @property
    def output_data_type(self) -> str:
        return "sector_price"
    
    @property
    def required_input_data(self) -> List[str]:
        return ["stock_price_data", "sector_definition"]
    
    def generate_from_csv(self, stock_data: Dict[str, pd.DataFrame], 
                         csv_path: str, output_path: str) -> pd.DataFrame:
        """CSV定義からセクターインデックスを生成"""
        try:
            sector_price_df, _ = self.sector_calculator.calc_sector_index(
                stock_data, csv_path, output_path
            )
            return sector_price_df
        except Exception as e:
            raise ValueError(f"セクターインデックスの生成に失敗しました: {e}")
    
    def generate_from_dict(self, stock_data: Dict[str, pd.DataFrame], 
                          sector_dict: Dict[str, List[str]]) -> pd.DataFrame:
        """辞書定義からセクターインデックスを生成"""
        try:
            # 時価総額データを事前計算
            marketcap_data = self.sector_calculator.calc_marketcap(
                stock_data['price'], stock_data['fin']
            )
            
            # セクターインデックスを生成
            sector_index = self.sector_calculator.calc_sector_index_by_dict(
                sector_dict, marketcap_data
            )
            
            return sector_index
        except Exception as e:
            raise ValueError(f"セクターインデックスの生成に失敗しました: {e}")
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """汎用生成メソッド"""
        if "sector_dict" in input_data:
            return self.generate_from_dict(
                input_data["stock_data"], 
                input_data["sector_dict"]
            )
        elif "csv_path" in input_data:
            return self.generate_from_csv(
                input_data["stock_data"],
                input_data["csv_path"],
                input_data.get("output_path", "temp_output.parquet")
            )
        else:
            raise ValueError("sector_dictまたはcsv_pathが必要です")


class CustomFactorIndexGenerator(DataGenerator):
    """カスタムファクターインデックス生成器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.sector_generator = SectorIndexGenerator()
    
    @property
    def output_data_type(self) -> str:
        return "factor_index"
    
    @property
    def required_input_data(self) -> List[str]:
        return ["stock_data", "factor_definition"]
    
    def generate_currency_sensitivity_index(self, stock_data: Dict[str, pd.DataFrame], 
                                          sensitivity_config: Dict[str, List[str]]) -> pd.DataFrame:
        """通貨感応度ベースのインデックス生成"""
        factor_dict = {}
        
        if "JPY_positive" in sensitivity_config:
            factor_dict["JPY+"] = sensitivity_config["JPY_positive"]
        if "JPY_negative" in sensitivity_config:
            factor_dict["JPY-"] = sensitivity_config["JPY_negative"]
        if "USD_positive" in sensitivity_config:
            factor_dict["USD+"] = sensitivity_config["USD_positive"]
        if "USD_negative" in sensitivity_config:
            factor_dict["USD-"] = sensitivity_config["USD_negative"]
        
        return self.sector_generator.generate_from_dict(stock_data, factor_dict)
    
    def generate_interest_rate_sensitivity_index(self, stock_data: Dict[str, pd.DataFrame], 
                                               rate_config: Dict[str, List[str]]) -> pd.DataFrame:
        """金利感応度ベースのインデックス生成"""
        factor_dict = {}
        
        if "rate_positive" in rate_config:
            factor_dict["Rate+"] = rate_config["rate_positive"]
        if "rate_negative" in rate_config:
            factor_dict["Rate-"] = rate_config["rate_negative"]
        
        return self.sector_generator.generate_from_dict(stock_data, factor_dict)
    
    def generate_size_factor_index(self, stock_data: Dict[str, pd.DataFrame], 
                                 size_config: Dict[str, Any]) -> pd.DataFrame:
        """サイズファクターベースのインデックス生成"""
        try:
            # 時価総額を計算
            marketcap_data = self.sector_generator.sector_calculator.calc_marketcap(
                stock_data['price'], stock_data['fin']
            )
            
            # 最新の時価総額でランキング
            latest_date = marketcap_data['Date'].max()
            latest_marketcap = marketcap_data[marketcap_data['Date'] == latest_date]
            
            # サイズ分位数で分割
            n_quantiles = size_config.get('n_quantiles', 3)
            quantile_labels = size_config.get('labels', [f'Size_Q{i+1}' for i in range(n_quantiles)])
            
            latest_marketcap['SizeQuantile'] = pd.qcut(
                latest_marketcap['MarketCapClose'], 
                q=n_quantiles, 
                labels=quantile_labels
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for label in quantile_labels:
                codes = latest_marketcap[latest_marketcap['SizeQuantile'] == label]['Code'].tolist()
                factor_dict[label] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"サイズファクターインデックスの生成に失敗しました: {e}")
    
    def generate_momentum_factor_index(self, stock_data: Dict[str, pd.DataFrame], 
                                     momentum_config: Dict[str, Any]) -> pd.DataFrame:
        """モメンタムファクターベースのインデックス生成"""
        try:
            # リターンを計算
            price_data = stock_data['price'].copy()
            lookback_days = momentum_config.get('lookback_days', 21)
            
            # セクター別にモメンタムを計算
            if 'Sector' in price_data.columns or 'Sector' in price_data.index.names:
                momentum_data = price_data.groupby('Code')['Close'].pct_change(lookback_days)
            else:
                momentum_data = price_data.groupby('Code')['Close'].pct_change(lookback_days)
            
            # 最新の期間でモメンタムランキング
            latest_momentum = momentum_data.groupby('Code').last()
            
            # モメンタム分位数で分割
            n_quantiles = momentum_config.get('n_quantiles', 3)
            quantile_labels = momentum_config.get('labels', [f'Mom_Q{i+1}' for i in range(n_quantiles)])
            
            momentum_quantiles = pd.qcut(
                latest_momentum,
                q=n_quantiles,
                labels=quantile_labels
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for label in quantile_labels:
                codes = momentum_quantiles[momentum_quantiles == label].index.tolist()
                factor_dict[label] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"モメンタムファクターインデックスの生成に失敗しました: {e}")
    
    def generate_value_factor_index(self, stock_data: Dict[str, pd.DataFrame],
                                  value_config: Dict[str, Any]) -> pd.DataFrame:
        """バリューファクターベースのインデックス生成"""
        try:
            # PBRやPERを計算
            price_data = stock_data['price']
            fin_data = stock_data['fin']
            
            # 最新の財務データと価格データを結合
            latest_price = price_data.groupby('Code').last().reset_index()
            latest_fin = fin_data.groupby('Code').last().reset_index()
            
            merged_data = pd.merge(latest_price, latest_fin, on='Code', how='inner')
            
            # バリュー指標を計算
            value_metric = value_config.get('metric', 'PBR')
            
            if value_metric == 'PBR':
                # PBR = 時価総額 / 純資産
                merged_data['Value_Score'] = merged_data['MarketCap'] / merged_data['NetAssets']
                ascending = True  # PBRは低いほど良い
            elif value_metric == 'PER':
                # PER = 時価総額 / 純利益
                merged_data['Value_Score'] = merged_data['MarketCap'] / merged_data['NetIncome']
                ascending = True  # PERは低いほど良い
            elif value_metric == 'Dividend_Yield':
                # 配当利回り
                merged_data['Value_Score'] = merged_data['DividendYield']
                ascending = False  # 配当利回りは高いほど良い
            else:
                raise ValueError(f"未対応のバリュー指標: {value_metric}")
            
            # バリュー分位数で分割
            n_quantiles = value_config.get('n_quantiles', 3)
            quantile_labels = value_config.get('labels', [f'Value_Q{i+1}' for i in range(n_quantiles)])
            
            merged_data['ValueQuantile'] = pd.qcut(
                merged_data['Value_Score'],
                q=n_quantiles,
                labels=quantile_labels
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for label in quantile_labels:
                codes = merged_data[merged_data['ValueQuantile'] == label]['Code'].tolist()
                factor_dict[label] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"バリューファクターインデックスの生成に失敗しました: {e}")
    
    def generate_quality_factor_index(self, stock_data: Dict[str, pd.DataFrame],
                                    quality_config: Dict[str, Any]) -> pd.DataFrame:
        """クオリティファクターベースのインデックス生成"""
        try:
            fin_data = stock_data['fin']
            
            # 最新の財務データ
            latest_fin = fin_data.groupby('Code').last().reset_index()
            
            # クオリティ指標を計算
            quality_metrics = quality_config.get('metrics', ['ROE', 'ROA', 'DebtEquityRatio'])
            weights = quality_config.get('weights', [0.4, 0.3, 0.3])
            
            quality_scores = {}
            
            for code in latest_fin['Code'].unique():
                code_data = latest_fin[latest_fin['Code'] == code].iloc[0]
                
                score = 0
                total_weight = 0
                
                for metric, weight in zip(quality_metrics, weights):
                    if metric in code_data and not pd.isna(code_data[metric]):
                        if metric == 'DebtEquityRatio':
                            # 負債比率は低いほど良い（逆転）
                            metric_score = 1 / (1 + code_data[metric])
                        else:
                            # ROE、ROAは高いほど良い
                            metric_score = code_data[metric]
                        
                        score += metric_score * weight
                        total_weight += weight
                
                if total_weight > 0:
                    quality_scores[code] = score / total_weight
            
            # クオリティスコアのDataFrameを作成
            quality_df = pd.DataFrame(
                list(quality_scores.items()),
                columns=['Code', 'Quality_Score']
            )
            
            # クオリティ分位数で分割
            n_quantiles = quality_config.get('n_quantiles', 3)
            quantile_labels = quality_config.get('labels', [f'Quality_Q{i+1}' for i in range(n_quantiles)])
            
            quality_df['QualityQuantile'] = pd.qcut(
                quality_df['Quality_Score'],
                q=n_quantiles,
                labels=quantile_labels
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for label in quantile_labels:
                codes = quality_df[quality_df['QualityQuantile'] == label]['Code'].tolist()
                factor_dict[label] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"クオリティファクターインデックスの生成に失敗しました: {e}")
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """汎用生成メソッド"""
        factor_type = input_data.get("factor_type", "custom")
        stock_data = input_data["stock_data"]
        
        if factor_type == "currency_sensitivity":
            return self.generate_currency_sensitivity_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "interest_rate_sensitivity":
            return self.generate_interest_rate_sensitivity_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "size_factor":
            return self.generate_size_factor_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "momentum_factor":
            return self.generate_momentum_factor_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "value_factor":
            return self.generate_value_factor_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "quality_factor":
            return self.generate_quality_factor_index(
                stock_data, input_data["factor_definition"]
            )
        elif factor_type == "custom":
            # 汎用的なカスタムファクター
            return self.sector_generator.generate_from_dict(
                stock_data, input_data["factor_definition"]
            )
        else:
            raise ValueError(f"未対応のファクタータイプ: {factor_type}")


class ThematicIndexGenerator(DataGenerator):
    """テーマ別インデックス生成器"""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.sector_generator = SectorIndexGenerator()
    
    @property
    def output_data_type(self) -> str:
        return "thematic_index"
    
    @property
    def required_input_data(self) -> List[str]:
        return ["stock_data", "theme_definition"]
    
    def generate_esg_index(self, stock_data: Dict[str, pd.DataFrame], 
                          esg_scores: Dict[str, float]) -> pd.DataFrame:
        """ESGスコアベースのインデックス生成"""
        try:
            # ESGスコアで分類
            esg_df = pd.DataFrame(list(esg_scores.items()), columns=['Code', 'ESG_Score'])
            
            # スコア分位数で分割
            n_quantiles = 3
            esg_df['ESG_Tier'] = pd.qcut(
                esg_df['ESG_Score'], 
                q=n_quantiles, 
                labels=['ESG_Low', 'ESG_Mid', 'ESG_High']
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for tier in ['ESG_Low', 'ESG_Mid', 'ESG_High']:
                codes = esg_df[esg_df['ESG_Tier'] == tier]['Code'].tolist()
                factor_dict[tier] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"ESGインデックスの生成に失敗しました: {e}")
    
    def generate_innovation_index(self, stock_data: Dict[str, pd.DataFrame], 
                                innovation_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """イノベーション指標ベースのインデックス生成"""
        try:
            # イノベーション指標を統合
            innovation_scores = {}
            
            for code, metrics in innovation_metrics.items():
                # 複数指標の加重平均
                weights = {
                    'r_and_d_ratio': 0.4,
                    'patent_count': 0.3,
                    'digital_transformation': 0.3
                }
                
                total_score = sum(
                    metrics.get(metric, 0) * weight 
                    for metric, weight in weights.items()
                )
                innovation_scores[code] = total_score
            
            # イノベーションスコアで分類
            innovation_df = pd.DataFrame(
                list(innovation_scores.items()), 
                columns=['Code', 'Innovation_Score']
            )
            
            # スコア分位数で分割
            innovation_df['Innovation_Tier'] = pd.qcut(
                innovation_df['Innovation_Score'], 
                q=3, 
                labels=['Innovation_Low', 'Innovation_Mid', 'Innovation_High']
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for tier in ['Innovation_Low', 'Innovation_Mid', 'Innovation_High']:
                codes = innovation_df[innovation_df['Innovation_Tier'] == tier]['Code'].tolist()
                factor_dict[tier] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"イノベーションインデックスの生成に失敗しました: {e}")
    
    def generate_dividend_yield_index(self, stock_data: Dict[str, pd.DataFrame], 
                                    dividend_data: Dict[str, float]) -> pd.DataFrame:
        """配当利回りベースのインデックス生成"""
        try:
            # 配当利回りで分類
            dividend_df = pd.DataFrame(
                list(dividend_data.items()), 
                columns=['Code', 'Dividend_Yield']
            )
            
            # 利回り分位数で分割
            dividend_df['Yield_Tier'] = pd.qcut(
                dividend_df['Dividend_Yield'], 
                q=4, 
                labels=['Yield_Q1', 'Yield_Q2', 'Yield_Q3', 'Yield_Q4']
            )
            
            # ファクター辞書を作成
            factor_dict = {}
            for tier in ['Yield_Q1', 'Yield_Q2', 'Yield_Q3', 'Yield_Q4']:
                codes = dividend_df[dividend_df['Yield_Tier'] == tier]['Code'].tolist()
                factor_dict[tier] = codes
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"配当利回りインデックスの生成に失敗しました: {e}")
    
    def generate_industry_4_0_index(self, stock_data: Dict[str, pd.DataFrame],
                                   industry_classification: Dict[str, List[str]]) -> pd.DataFrame:
        """インダストリー4.0関連インデックス生成"""
        try:
            # インダストリー4.0関連分野
            industry_4_0_sectors = {
                'AI_Robotics': industry_classification.get('ai_robotics', []),
                'IoT_Sensors': industry_classification.get('iot_sensors', []),
                'Cloud_Computing': industry_classification.get('cloud_computing', []),
                'Manufacturing_Tech': industry_classification.get('manufacturing_tech', []),
                'Data_Analytics': industry_classification.get('data_analytics', [])
            }
            
            # 空のリストを除外
            factor_dict = {k: v for k, v in industry_4_0_sectors.items() if v}
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"インダストリー4.0インデックスの生成に失敗しました: {e}")
    
    def generate_renewable_energy_index(self, stock_data: Dict[str, pd.DataFrame],
                                      energy_classification: Dict[str, List[str]]) -> pd.DataFrame:
        """再生可能エネルギーインデックス生成"""
        try:
            # 再生可能エネルギー関連分野
            renewable_sectors = {
                'Solar_Power': energy_classification.get('solar', []),
                'Wind_Power': energy_classification.get('wind', []),
                'Hydro_Power': energy_classification.get('hydro', []),
                'Battery_Storage': energy_classification.get('battery', []),
                'Grid_Infrastructure': energy_classification.get('grid', [])
            }
            
            # 空のリストを除外
            factor_dict = {k: v for k, v in renewable_sectors.items() if v}
            
            return self.sector_generator.generate_from_dict(stock_data, factor_dict)
            
        except Exception as e:
            raise ValueError(f"再生可能エネルギーインデックスの生成に失敗しました: {e}")
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """汎用生成メソッド"""
        theme_type = input_data.get("theme_type", "custom")
        stock_data = input_data["stock_data"]
        
        if theme_type == "esg":
            return self.generate_esg_index(
                stock_data, input_data["theme_definition"]
            )
        elif theme_type == "innovation":
            return self.generate_innovation_index(
                stock_data, input_data["theme_definition"]
            )
        elif theme_type == "dividend_yield":
            return self.generate_dividend_yield_index(
                stock_data, input_data["theme_definition"]
            )
        elif theme_type == "industry_4_0":
            return self.generate_industry_4_0_index(
                stock_data, input_data["theme_definition"]
            )
        elif theme_type == "renewable_energy":
            return self.generate_renewable_energy_index(
                stock_data, input_data["theme_definition"]
            )
        else:
            raise ValueError(f"未対応のテーマタイプ: {theme_type}")


class SyntheticDataGenerator(DataGenerator):
    """合成データ生成器（テスト・検証用）"""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    @property
    def output_data_type(self) -> str:
        return "synthetic_data"
    
    @property
    def required_input_data(self) -> List[str]:
        return ["config"]
    
    def generate_random_walk_prices(self, config: Dict[str, Any]) -> pd.DataFrame:
        """ランダムウォーク価格データを生成"""
        n_days = config.get('n_days', 1000)
        n_sectors = config.get('n_sectors', 5)
        initial_price = config.get('initial_price', 100)
        volatility = config.get('volatility', 0.02)
        drift = config.get('drift', 0.0005)
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        # セクター名を作成
        sectors = config.get('sector_names', [f'Sector_{i+1}' for i in range(n_sectors)])
        
        # データを生成
        np.random.seed(config.get('random_seed', 42))
        data = []
        
        for sector in sectors:
            # ランダムウォークでリターンを生成
            returns = np.random.normal(drift, volatility, n_days)
            prices = initial_price * np.exp(np.cumsum(returns))
            
            for i, date in enumerate(dates):
                # OHLC価格を生成
                close = prices[i]
                daily_range = close * np.random.uniform(0.005, 0.03)
                high = close + np.random.uniform(0, daily_range)
                low = close - np.random.uniform(0, daily_range)
                open_price = low + np.random.uniform(0, high - low)
                
                data.append({
                    'Date': date,
                    'Sector': sector,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': close,
                    'Volume': np.random.uniform(1000000, 10000000)
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['Date', 'Sector'])
    
    def generate_correlated_factors(self, config: Dict[str, Any]) -> pd.DataFrame:
        """相関のあるファクターデータを生成"""
        n_days = config.get('n_days', 1000)
        n_factors = config.get('n_factors', 3)
        correlation_matrix = config.get('correlation_matrix', None)
        factor_names = config.get('factor_names', [f'Factor_{i+1}' for i in range(n_factors)])
        
        # デフォルトの相関行列
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_factors)
            correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.3
            correlation_matrix[0, 2] = correlation_matrix[2, 0] = -0.2
            correlation_matrix[1, 2] = correlation_matrix[2, 1] = 0.1
        
        # 多変量正規分布から生成
        np.random.seed(config.get('random_seed', 42))
        mean = np.zeros(n_factors)
        factors = np.random.multivariate_normal(mean, correlation_matrix, n_days)
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        # データフレームを作成
        factor_df = pd.DataFrame(factors, columns=factor_names, index=dates)
        factor_df.index.name = 'Date'
        
        return factor_df
    
    def generate_regime_switching_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """レジーム・スイッチングデータを生成"""
        n_days = config.get('n_days', 1000)
        regimes = config.get('regimes', {
            'bull': {'prob': 0.6, 'mean': 0.001, 'vol': 0.015},
            'bear': {'prob': 0.3, 'mean': -0.0005, 'vol': 0.025},
            'neutral': {'prob': 0.1, 'mean': 0.0, 'vol': 0.02}
        })
        
        # レジーム遷移を生成
        np.random.seed(config.get('random_seed', 42))
        regime_names = list(regimes.keys())
        regime_probs = [regimes[name]['prob'] for name in regime_names]
        
        current_regime = np.random.choice(regime_names, p=regime_probs)
        regime_sequence = [current_regime]
        
        # マルコフ連鎖でレジーム遷移
        transition_prob = config.get('regime_persistence', 0.95)
        
        for _ in range(n_days - 1):
            if np.random.random() < transition_prob:
                # 同じレジームを維持
                regime_sequence.append(current_regime)
            else:
                # レジーム変更
                current_regime = np.random.choice(regime_names, p=regime_probs)
                regime_sequence.append(current_regime)
        
        # 各レジームに応じたリターンを生成
        returns = []
        for regime in regime_sequence:
            regime_config = regimes[regime]
            return_val = np.random.normal(regime_config['mean'], regime_config['vol'])
            returns.append(return_val)
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        # データフレームを作成
        data_df = pd.DataFrame({
            'Date': dates,
            'Returns': returns,
            'Regime': regime_sequence
        })
        
        # 価格系列を計算
        data_df['Price'] = 100 * np.exp(np.cumsum(data_df['Returns']))
        
        return data_df.set_index('Date')
    
    def generate_factor_model_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """ファクターモデルに基づくデータを生成"""
        n_days = config.get('n_days', 1000)
        n_assets = config.get('n_assets', 10)
        n_factors = config.get('n_factors', 3)
        
        # ファクター名
        factor_names = config.get('factor_names', [f'Factor_{i+1}' for i in range(n_factors)])
        asset_names = config.get('asset_names', [f'Asset_{i+1}' for i in range(n_assets)])
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        np.random.seed(config.get('random_seed', 42))
        
        # ファクターリターンを生成
        factor_vol = config.get('factor_volatility', 0.02)
        factors = np.random.normal(0, factor_vol, (n_days, n_factors))
        
        # ファクターローディング（ベータ）を生成
        factor_loadings = np.random.normal(0, 1, (n_assets, n_factors))
        
        # 固有リスク（idiosyncratic risk）を生成
        idio_vol = config.get('idiosyncratic_volatility', 0.03)
        idiosyncratic_returns = np.random.normal(0, idio_vol, (n_days, n_assets))
        
        # 資産リターンを計算: R = Beta * F + epsilon
        asset_returns = np.dot(factors, factor_loadings.T) + idiosyncratic_returns
        
        # データフレームを作成
        data = []
        for i, date in enumerate(dates):
            for j, asset in enumerate(asset_names):
                data.append({
                    'Date': date,
                    'Asset': asset,
                    'Return': asset_returns[i, j]
                })
                # ファクターデータも含める
                for k, factor in enumerate(factor_names):
                    data[-1][factor] = factors[i, k]
        
        df = pd.DataFrame(data)
        return df.set_index(['Date', 'Asset'])
    
    def generate_jump_diffusion_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """ジャンプ拡散モデルのデータを生成"""
        n_days = config.get('n_days', 1000)
        n_assets = config.get('n_assets', 5)
        
        # パラメータ
        drift = config.get('drift', 0.0005)
        volatility = config.get('volatility', 0.02)
        jump_intensity = config.get('jump_intensity', 0.05)  # 1日あたりのジャンプ確率
        jump_mean = config.get('jump_mean', 0.0)
        jump_std = config.get('jump_std', 0.05)
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        asset_names = config.get('asset_names', [f'Asset_{i+1}' for i in range(n_assets)])
        
        np.random.seed(config.get('random_seed', 42))
        
        data = []
        
        for asset in asset_names:
            # 拡散項（ブラウン運動）
            diffusion_returns = np.random.normal(drift, volatility, n_days)
            
            # ジャンプ項
            jump_times = np.random.poisson(jump_intensity, n_days) > 0
            jump_sizes = np.random.normal(jump_mean, jump_std, n_days)
            jump_returns = jump_times * jump_sizes
            
            # 総リターン
            total_returns = diffusion_returns + jump_returns
            
            # 価格を計算
            prices = 100 * np.exp(np.cumsum(total_returns))
            
            for i, date in enumerate(dates):
                data.append({
                    'Date': date,
                    'Asset': asset,
                    'Price': prices[i],
                    'Return': total_returns[i],
                    'Diffusion_Return': diffusion_returns[i],
                    'Jump_Return': jump_returns[i],
                    'Has_Jump': jump_times[i]
                })
        
        df = pd.DataFrame(data)
        return df.set_index(['Date', 'Asset'])
    
    def generate_garch_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """GARCH モデルのデータを生成"""
        n_days = config.get('n_days', 1000)
        
        # GARCH パラメータ
        omega = config.get('omega', 0.00001)  # 定数項
        alpha = config.get('alpha', 0.1)      # ARCH項
        beta = config.get('beta', 0.85)       # GARCH項
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        np.random.seed(config.get('random_seed', 42))
        
        # 初期値
        returns = np.zeros(n_days)
        volatilities = np.zeros(n_days)
        volatilities[0] = np.sqrt(omega / (1 - alpha - beta))  # 無条件分散
        
        # GARCH プロセスを生成
        for t in range(1, n_days):
            # 標準化された残差
            z_t = np.random.normal(0, 1)
            
            # 条件付き分散
            volatilities[t] = np.sqrt(
                omega + alpha * returns[t-1]**2 + beta * volatilities[t-1]**2
            )
            
            # リターン
            returns[t] = volatilities[t] * z_t
        
        # データフレームを作成
        df = pd.DataFrame({
            'Date': dates,
            'Return': returns,
            'Volatility': volatilities,
            'Price': 100 * np.exp(np.cumsum(returns))
        })
        
        return df.set_index('Date')
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """汎用生成メソッド"""
        data_type = input_data.get("data_type", "random_walk")
        config = input_data.get("config", {})
        
        if data_type == "random_walk":
            return self.generate_random_walk_prices(config)
        elif data_type == "correlated_factors":
            return self.generate_correlated_factors(config)
        elif data_type == "regime_switching":
            return self.generate_regime_switching_data(config)
        elif data_type == "factor_model":
            return self.generate_factor_model_data(config)
        elif data_type == "jump_diffusion":
            return self.generate_jump_diffusion_data(config)
        elif data_type == "garch":
            return self.generate_garch_data(config)
        else:
            raise ValueError(f"未対応の合成データタイプ: {data_type}")


class MarketDataGenerator(DataGenerator):
    """マーケットデータ生成器（実市場データの模倣）"""
    
    def __init__(self, **kwargs):
        super().__init__()
    
    @property
    def output_data_type(self) -> str:
        return "market_data"
    
    @property
    def required_input_data(self) -> List[str]:
        return ["market_config"]
    
    def generate_market_regime_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """市場レジームに基づくデータを生成"""
        n_days = config.get('n_days', 1000)
        n_sectors = config.get('n_sectors', 10)
        
        # 市場レジーム定義
        market_regimes = {
            'bull_market': {
                'prob': 0.4,
                'market_return': 0.0008,
                'market_vol': 0.015,
                'sector_correlation': 0.7
            },
            'bear_market': {
                'prob': 0.2,
                'market_return': -0.0005,
                'market_vol': 0.025,
                'sector_correlation': 0.9
            },
            'sideways_market': {
                'prob': 0.4,
                'market_return': 0.0002,
                'market_vol': 0.018,
                'sector_correlation': 0.5
            }
        }
        
        # 日付範囲を作成
        dates = pd.date_range(
            start=config.get('start_date', '2020-01-01'),
            periods=n_days,
            freq='B'
        )
        
        sector_names = config.get('sector_names', [f'Sector_{i+1}' for i in range(n_sectors)])
        
        np.random.seed(config.get('random_seed', 42))
        
        # レジーム遷移を生成
        regime_names = list(market_regimes.keys())
        regime_probs = [market_regimes[name]['prob'] for name in regime_names]
        
        current_regime = np.random.choice(regime_names, p=regime_probs)
        regimes = [current_regime]
        
        transition_prob = 0.98  # 高い持続性
        
        for _ in range(n_days - 1):
            if np.random.random() < transition_prob:
                regimes.append(current_regime)
            else:
                current_regime = np.random.choice(regime_names, p=regime_probs)
                regimes.append(current_regime)
        
        # データを生成
        data = []
        
        for i, date in enumerate(dates):
            regime = regimes[i]
            regime_config = market_regimes[regime]
            
            # マーケットファクター
            market_return = np.random.normal(
                regime_config['market_return'],
                regime_config['market_vol']
            )
            
            # セクター固有リターン
            correlation = regime_config['sector_correlation']
            sector_vol = 0.02
            
            for sector in sector_names:
                # 相関を考慮したセクターリターン
                idiosyncratic = np.random.normal(0, sector_vol * np.sqrt(1 - correlation**2))
                sector_return = correlation * market_return + idiosyncratic
                
                data.append({
                    'Date': date,
                    'Sector': sector,
                    'Return': sector_return,
                    'Market_Return': market_return,
                    'Regime': regime,
                    'Sector_Specific': idiosyncratic
                })
        
        df = pd.DataFrame(data)
        
        # 価格を計算
        df = df.sort_values(['Sector', 'Date'])
        df['Price'] = df.groupby('Sector')['Return'].apply(
            lambda x: 100 * np.exp(x.cumsum())
        ).reset_index(drop=True)
        
        return df.set_index(['Date', 'Sector'])
    
    def generate(self, input_data: Dict[str, Any], **kwargs) -> pd.DataFrame:
        """汎用生成メソッド"""
        config = input_data.get("market_config", {})
        return self.generate_market_regime_data(config)


# ファクトリー関数
def get_data_generator(generator_type: str, **kwargs) -> DataGenerator:
    """データ生成器タイプに応じた生成器を取得"""
    generators = {
        'sector_index': SectorIndexGenerator,
        'custom_factor': CustomFactorIndexGenerator,
        'thematic_index': ThematicIndexGenerator,
        'synthetic_data': SyntheticDataGenerator,
        'market_data': MarketDataGenerator
    }
    
    if generator_type not in generators:
        raise ValueError(f"未対応のデータ生成器タイプです: {generator_type}")
    
    generator_class = generators[generator_type]
    return generator_class(**kwargs)


# 便利関数
def create_currency_factor_data(stock_data: Dict[str, pd.DataFrame], 
                              jpy_positive_codes: List[str],
                              jpy_negative_codes: List[str]) -> pd.DataFrame:
    """通貨ファクターデータを簡単に作成"""
    generator = CustomFactorIndexGenerator()
    sensitivity_config = {
        "JPY_positive": jpy_positive_codes,
        "JPY_negative": jpy_negative_codes
    }
    return generator.generate_currency_sensitivity_index(stock_data, sensitivity_config)


def create_size_factor_data(stock_data: Dict[str, pd.DataFrame], 
                           n_quantiles: int = 3) -> pd.DataFrame:
    """サイズファクターデータを簡単に作成"""
    generator = CustomFactorIndexGenerator()
    size_config = {
        "n_quantiles": n_quantiles,
        "labels": [f"Size_Q{i+1}" for i in range(n_quantiles)]
    }
    input_data = {
        "stock_data": stock_data,
        "factor_type": "size_factor",
        "factor_definition": size_config
    }
    return generator.generate(input_data)


def create_value_factor_data(stock_data: Dict[str, pd.DataFrame], 
                           metric: str = 'PBR', 
                           n_quantiles: int = 3) -> pd.DataFrame:
    """バリューファクターデータを簡単に作成"""
    generator = CustomFactorIndexGenerator()
    value_config = {
        "metric": metric,
        "n_quantiles": n_quantiles,
        "labels": [f"Value_Q{i+1}" for i in range(n_quantiles)]
    }
    input_data = {
        "stock_data": stock_data,
        "factor_type": "value_factor",
        "factor_definition": value_config
    }
    return generator.generate(input_data)


def create_momentum_factor_data(stock_data: Dict[str, pd.DataFrame], 
                              lookback_days: int = 21,
                              n_quantiles: int = 3) -> pd.DataFrame:
    """モメンタムファクターデータを簡単に作成"""
    generator = CustomFactorIndexGenerator()
    momentum_config = {
        "lookback_days": lookback_days,
        "n_quantiles": n_quantiles,
        "labels": [f"Mom_Q{i+1}" for i in range(n_quantiles)]
    }
    input_data = {
        "stock_data": stock_data,
        "factor_type": "momentum_factor",
        "factor_definition": momentum_config
    }
    return generator.generate(input_data)


def create_test_data(n_days: int = 1000, n_sectors: int = 5, 
                    random_seed: int = 42) -> pd.DataFrame:
    """テスト用データを簡単に作成"""
    generator = SyntheticDataGenerator()
    config = {
        "n_days": n_days,
        "n_sectors": n_sectors,
        "random_seed": random_seed,
        "volatility": 0.02,
        "drift": 0.0005
    }
    input_data = {
        "data_type": "random_walk",
        "config": config
    }
    return generator.generate(input_data)


def create_market_simulation_data(n_days: int = 1000, n_sectors: int = 10,
                                 random_seed: int = 42) -> pd.DataFrame:
    """市場シミュレーションデータを作成"""
    generator = MarketDataGenerator()
    config = {
        "n_days": n_days,
        "n_sectors": n_sectors,
        "random_seed": random_seed
    }
    input_data = {
        "market_config": config
    }
    return generator.generate(input_data)


def create_factor_model_simulation(n_days: int = 1000, n_assets: int = 20,
                                 n_factors: int = 3, random_seed: int = 42) -> pd.DataFrame:
    """ファクターモデルシミュレーションデータを作成"""
    generator = SyntheticDataGenerator()
    config = {
        "n_days": n_days,
        "n_assets": n_assets,
        "n_factors": n_factors,
        "random_seed": random_seed,
        "factor_volatility": 0.02,
        "idiosyncratic_volatility": 0.03
    }
    input_data = {
        "data_type": "factor_model",
        "config": config
    }
    return generator.generate(input_data)


# 高度な組み合わせ関数
def create_multi_factor_universe(stock_data: Dict[str, pd.DataFrame],
                                factor_configs: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """複数ファクターのユニバースを作成"""
    generator = CustomFactorIndexGenerator()
    factor_data = {}
    
    for factor_name, config in factor_configs.items():
        try:
            input_data = {
                "stock_data": stock_data,
                "factor_type": config["type"],
                "factor_definition": config["definition"]
            }
            factor_data[factor_name] = generator.generate(input_data)
        except Exception as e:
            warnings.warn(f"ファクター '{factor_name}' の生成に失敗しました: {e}")
    
    return factor_data


def create_thematic_universe(stock_data: Dict[str, pd.DataFrame],
                           theme_configs: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
    """テーマ別ユニバースを作成"""
    generator = ThematicIndexGenerator()
    theme_data = {}
    
    for theme_name, config in theme_configs.items():
        try:
            input_data = {
                "stock_data": stock_data,
                "theme_type": config["type"],
                "theme_definition": config["definition"]
            }
            theme_data[theme_name] = generator.generate(input_data)
        except Exception as e:
            warnings.warn(f"テーマ '{theme_name}' の生成に失敗しました: {e}")
    
    return theme_data