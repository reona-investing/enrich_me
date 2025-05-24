# %% [markdown]
# # データ処理システム検証・修正ノートブック
# 
# このノートブックでは、新しいリファクタリング済みデータ処理システムの検証と、
# 既存システムとの互換性確保のための修正を行います。

# %%
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Dict, Any, List
import sys
import os

# パスの設定
sys.path.append('../../..')
from utils.paths import Paths

# %%
# 必要なモジュールのインポート
try:
    from data_processing.core.return_calculators import get_return_calculator
    from data_processing.core.transformers.financial_transformers import (
        MomentumTransformer, VolatilityTransformer, RankingTransformer, SectorCategoricalTransformer
    )
    from data_processing.core.pipeline.core_pipeline import CalculationPipeline
    from data_processing.facades.legacy_features_facade import LegacyFeaturesFacade
    print("✓ 新システムのモジュールインポート成功")
except ImportError as e:
    print(f"✗ 新システムのモジュールインポートエラー: {e}")

try:
    from calculation.features_calculator import FeaturesCalculator
    print("✓ 既存システムのモジュールインポート成功")
except ImportError as e:
    print(f"✗ 既存システムのモジュールインポートエラー: {e}")

# %% [markdown]
# ## 1. サンプルデータの作成と検証

# %%
def create_sample_sector_data(n_days=1000, n_sectors=5):
    """検証用のサンプルセクターデータを作成"""
    # 日付範囲を作成
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    sectors = ['Technology', 'Financials', 'Healthcare', 'Energy', 'Materials'][:n_sectors]
    
    # セクター価格データを作成
    np.random.seed(42)
    sector_data = []
    
    for sector in sectors:
        # ランダムウォークで価格を生成
        returns = np.random.normal(0.0005, 0.02, n_days)  
        prices = 100 * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            # OHLC価格を生成
            close = prices[i]
            daily_range = close * np.random.uniform(0.005, 0.03)
            high = close + np.random.uniform(0, daily_range)
            low = close - np.random.uniform(0, daily_range)
            open_price = low + np.random.uniform(0, high - low)
            
            sector_data.append({
                'Date': date,
                'Sector': sector,
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': np.random.uniform(1000000, 10000000)
            })
    
    sector_price_df = pd.DataFrame(sector_data)
    sector_price_df = sector_price_df.set_index(['Date', 'Sector'])
    
    # セクターリストを作成
    sector_list_data = []
    for i, sector in enumerate(sectors):
        n_stocks = np.random.randint(3, 6)
        for j in range(n_stocks):
            sector_list_data.append({
                'Code': f'{1000 + i*100 + j}',
                'Sector': sector,
                'Name': f'{sector}_Stock_{j+1}'
            })
    
    sector_list_df = pd.DataFrame(sector_list_data)
    
    # 株価データ辞書（簡易版）
    stock_dfs_dict = {
        'price': pd.DataFrame({
            'Date': dates[:100],
            'Code': '1000',
            'Close': 100 + np.random.randn(100).cumsum()
        }),
        'fin': pd.DataFrame({
            'Date': dates[::30][:12],
            'Code': '1000',
            'ForecastEPS': 10 + np.random.randn(12)
        })
    }
    
    return sector_price_df, sector_list_df, stock_dfs_dict

# サンプルデータを作成
print("サンプルデータを作成中...")
sector_price_df, sector_list_df, stock_dfs_dict = create_sample_sector_data()

print(f"セクター価格データ形状: {sector_price_df.shape}")
print(f"セクター定義データ形状: {sector_list_df.shape}")
print(f"インデックス構造: {sector_price_df.index.names}")
print("\nセクター価格データのサンプル:")
print(sector_price_df.head())

# %% [markdown]
# ## 2. データ前処理の検証

# %%
def preprocess_sector_data(sector_price_df):
    """セクター価格データの前処理"""
    print("データ前処理を実行中...")
    
    # 1日リターンを計算
    processed_df = sector_price_df.copy()
    processed_df['1d_return'] = processed_df.groupby(level='Sector')['Close'].pct_change(1)
    
    print(f"1d_returnの統計:")
    print(f"  欠損値数: {processed_df['1d_return'].isna().sum()}")
    print(f"  有効値数: {processed_df['1d_return'].notna().sum()}")
    print(f"  平均: {processed_df['1d_return'].mean():.6f}")
    print(f"  標準偏差: {processed_df['1d_return'].std():.6f}")
    
    return processed_df

# データを前処理
processed_sector_df = preprocess_sector_data(sector_price_df)

# %% [markdown]
# ## 3. 新システムのコンポーネント単体テスト

# %%
def test_momentum_transformer():
    """モメンタム変換器の単体テスト"""
    print("\n=== モメンタム変換器テスト ===")
    
    try:
        # モメンタム変換器を作成
        momentum_transformer = MomentumTransformer(
            windows=[5, 21],
            base_column='1d_return',
            add_rank=True,
            exclude_current_day=True
        )
        
        # 変換を実行
        result = momentum_transformer.execute(processed_sector_df)
        
        print(f"✓ モメンタム変換成功")
        print(f"  出力形状: {result.shape}")
        print(f"  出力カラム: {list(result.columns)}")
        print(f"  モメンタムカラムの欠損値:")
        for col in result.columns:
            if 'mom' in col:
                print(f"    {col}: {result[col].isna().sum()}")
        
        return result
        
    except Exception as e:
        print(f"✗ モメンタム変換器エラー: {e}")
        return None

def test_volatility_transformer():
    """ボラティリティ変換器の単体テスト"""
    print("\n=== ボラティリティ変換器テスト ===")
    
    try:
        # ボラティリティ変換器を作成
        volatility_transformer = VolatilityTransformer(
            windows=[5, 21],
            base_column='1d_return',
            add_rank=True,
            exclude_current_day=True
        )
        
        # 変換を実行
        result = volatility_transformer.execute(processed_sector_df)
        
        print(f"✓ ボラティリティ変換成功")
        print(f"  出力形状: {result.shape}")
        print(f"  出力カラム: {list(result.columns)}")
        print(f"  ボラティリティカラムの欠損値:")
        for col in result.columns:
            if 'vola' in col:
                print(f"    {col}: {result[col].isna().sum()}")
        
        return result
        
    except Exception as e:
        print(f"✗ ボラティリティ変換器エラー: {e}")
        return None

def test_ranking_transformer():
    """ランキング変換器の単体テスト"""
    print("\n=== ランキング変換器テスト ===")
    
    try:
        # 事前にモメンタムカラムを作成
        momentum_result = test_momentum_transformer()
        if momentum_result is None:
            print("✗ 前提となるモメンタム変換が失敗したため、ランキングテストをスキップ")
            return None
        
        # ランキング変換器を作成
        ranking_transformer = RankingTransformer(
            target_columns=['5d_mom', '21d_mom'],
            ranking_method='dense',
            ascending=False
        )
        
        # 変換を実行
        result = ranking_transformer.execute(momentum_result)
        
        print(f"✓ ランキング変換成功")
        print(f"  出力形状: {result.shape}")
        print(f"  新規ランキングカラム:")
        ranking_cols = [col for col in result.columns if col.endswith('_rank') and col not in momentum_result.columns]
        for col in ranking_cols:
            print(f"    {col}: 範囲[{result[col].min():.0f}, {result[col].max():.0f}]")
        
        return result
        
    except Exception as e:
        print(f"✗ ランキング変換器エラー: {e}")
        return None

def test_sector_categorical_transformer():
    """セクターカテゴリ変換器の単体テスト"""
    print("\n=== セクターカテゴリ変換器テスト ===")
    
    try:
        # セクターカテゴリ変換器を作成
        sector_transformer = SectorCategoricalTransformer()
        
        # 変換を実行
        result = sector_transformer.execute(processed_sector_df)
        
        print(f"✓ セクターカテゴリ変換成功")
        print(f"  出力形状: {result.shape}")
        if 'Sector_cat' in result.columns:
            print(f"  セクターカテゴリ範囲: [{result['Sector_cat'].min()}, {result['Sector_cat'].max()}]")
            print(f"  ユニークセクター数: {result['Sector_cat'].nunique()}")
        
        return result
        
    except Exception as e:
        print(f"✗ セクターカテゴリ変換器エラー: {e}")
        return None

# 各変換器をテスト
momentum_result = test_momentum_transformer()
volatility_result = test_volatility_transformer()
ranking_result = test_ranking_transformer()
sector_cat_result = test_sector_categorical_transformer()

# %% [markdown]
# ## 4. パイプライン統合テスト

# %%
def test_integrated_pipeline():
    """統合パイプラインのテスト"""
    print("\n=== 統合パイプラインテスト ===")
    
    try:
        # パイプラインを構築
        pipeline = CalculationPipeline()
        
        # 変換器を追加
        pipeline.add_transformer(MomentumTransformer(
            windows=[5, 21],
            base_column='1d_return',
            add_rank=True,
            exclude_current_day=True
        ))
        
        pipeline.add_transformer(VolatilityTransformer(
            windows=[5, 21],
            base_column='1d_return',
            add_rank=True,
            exclude_current_day=True
        ))
        
        pipeline.add_transformer(SectorCategoricalTransformer())
        
        # パイプライン構成を検証
        validation = pipeline.validate_pipeline()
        print(f"パイプライン検証結果: {validation['is_valid']}")
        if validation['warnings']:
            print(f"警告: {validation['warnings']}")
        
        # パイプラインを実行
        result = pipeline.execute(processed_sector_df)
        
        print(f"✓ 統合パイプライン成功")
        print(f"  入力形状: {processed_sector_df.shape}")
        print(f"  出力形状: {result.shape}")
        print(f"  出力カラム数: {len(result.columns)}")
        print(f"  実行時間: {pipeline.get_execution_metadata()['execution_time']:.3f}秒")
        
        # カラム別の欠損値チェック
        print(f"\n  カラム別欠損値:")
        for col in result.columns:
            missing_count = result[col].isna().sum()
            if missing_count > 0:
                print(f"    {col}: {missing_count}")
        
        return result
        
    except Exception as e:
        print(f"✗ 統合パイプラインエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

# 統合パイプラインをテスト
integrated_result = test_integrated_pipeline()

# %% [markdown]
# ## 5. 既存システムとの比較検証

# %%
def test_legacy_system():
    """既存システムの動作テスト（簡易版）"""
    print("\n=== 既存システムテスト ===")
    
    try:
        # 既存システムで計算（インデックス系を無効化）
        legacy_result = FeaturesCalculator.calculate_features(
            new_sector_price=sector_price_df,
            new_sector_list=sector_list_df,
            stock_dfs_dict=stock_dfs_dict,
            adopts_features_indices=False,  # 簡易テストのため無効化
            adopts_features_price=True,
            adopt_1d_return=True,
            mom_duration=[5, 21],
            vola_duration=[5, 21],
            adopt_size_factor=False,  # 簡易テストのため無効化
            adopt_eps_factor=False,   # 簡易テストのため無効化
            adopt_sector_categorical=True,
            add_rank=True
        )
        
        print(f"✓ 既存システム成功")
        print(f"  出力形状: {legacy_result.shape}")
        print(f"  出力カラム: {list(legacy_result.columns)}")
        
        return legacy_result
        
    except Exception as e:
        print(f"✗ 既存システムエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_new_vs_legacy():
    """新旧システムの比較"""
    print("\n=== 新旧システム比較 ===")
    
    # 新システムの結果
    new_result = integrated_result
    if new_result is None:
        print("✗ 新システムの結果がないため比較できません")
        return
    
    # 既存システムの結果  
    legacy_result = test_legacy_system()
    if legacy_result is None:
        print("✗ 既存システムの結果がないため比較できません")
        return
    
    # 共通カラムで比較
    common_columns = set(new_result.columns) & set(legacy_result.columns)
    print(f"比較対象カラム数: {len(common_columns)}")
    
    if len(common_columns) == 0:
        print("✗ 共通カラムがありません")
        print(f"新システムカラム: {list(new_result.columns)}")
        print(f"既存システムカラム: {list(legacy_result.columns)}")
        return
    
    # カラム別比較
    comparison_results = {}
    for col in common_columns:
        try:
            # 共通インデックスで比較
            common_index = new_result.index.intersection(legacy_result.index)
            if len(common_index) == 0:
                continue
                
            new_series = new_result.loc[common_index, col]
            legacy_series = legacy_result.loc[common_index, col]
            
            # 有効値のみで比較
            valid_mask = new_series.notna() & legacy_series.notna()
            if valid_mask.sum() < 10:  # 最低10個のデータポイントが必要
                continue
                
            new_valid = new_series[valid_mask]
            legacy_valid = legacy_series[valid_mask]
            
            # 統計的比較
            correlation = new_valid.corr(legacy_valid)
            max_diff = (new_valid - legacy_valid).abs().max()
            mean_diff = (new_valid - legacy_valid).abs().mean()
            
            comparison_results[col] = {
                'correlation': correlation,
                'max_absolute_difference': max_diff,
                'mean_absolute_difference': mean_diff,
                'common_data_points': len(new_valid)
            }
            
        except Exception as e:
            print(f"カラム '{col}' の比較でエラー: {e}")
    
    # 結果を表示
    print(f"\n比較完了カラム数: {len(comparison_results)}")
    
    if comparison_results:
        print("\n詳細比較結果:")
        for col, stats in list(comparison_results.items())[:10]:  # 上位10件
            corr = stats['correlation']
            max_diff = stats['max_absolute_difference']
            data_points = stats['common_data_points']
            
            if not pd.isna(corr):
                status = "✓" if corr > 0.95 else "⚠" if corr > 0.80 else "✗"
                print(f"  {status} {col}: 相関={corr:.4f}, 最大差={max_diff:.6f}, データ点数={data_points}")
    
    return comparison_results

# 比較を実行
comparison_results = compare_new_vs_legacy()

# %% [markdown]
# ## 6. 修正されたファサードのテスト

# %%
def create_fixed_legacy_facade():
    """修正版のLegacyFeaturesFacadeを作成"""
    
    class FixedLegacyFeaturesFacade(LegacyFeaturesFacade):
        """修正版のLegacyFeaturesFacade"""
        
        def _calculate_features_new_system(self, **kwargs) -> pd.DataFrame:
            """修正版の新システム特徴量算出"""
            
            # パイプラインを構築
            pipeline = CalculationPipeline()
            
            # 価格系特徴量の設定
            if kwargs.get('adopts_features_price', True):
                # 事前に1日リターンを計算
                sector_price_df = kwargs['new_sector_price'].copy()
                if '1d_return' not in sector_price_df.columns:
                    sector_price_df['1d_return'] = sector_price_df.groupby(level='Sector')['Close'].pct_change(1)
                
                # モメンタム変換
                if kwargs.get('mom_duration'):
                    momentum_transformer = MomentumTransformer(
                        windows=kwargs['mom_duration'],
                        base_column='1d_return',
                        add_rank=kwargs.get('add_rank', True),
                        exclude_current_day=True
                    )
                    pipeline.add_transformer(momentum_transformer)
                
                # ボラティリティ変換
                if kwargs.get('vola_duration'):
                    volatility_transformer = VolatilityTransformer(
                        windows=kwargs['vola_duration'],
                        base_column='1d_return',
                        add_rank=kwargs.get('add_rank', True),
                        exclude_current_day=True
                    )
                    pipeline.add_transformer(volatility_transformer)
                
                # セクターカテゴリ変換
                if kwargs.get('adopt_sector_categorical', True):
                    pipeline.add_transformer(SectorCategoricalTransformer())
            else:
                sector_price_df = kwargs['new_sector_price']
            
            # 新システムでパイプライン実行
            if len(pipeline.transformers) > 0:
                price_df = pipeline.execute(sector_price_df)
            else:
                price_df = sector_price_df.copy()
            
            # 基本的な1d_returnを含める
            if 'adopt_1d_return' in kwargs and kwargs['adopt_1d_return']:
                if '1d_return' not in price_df.columns and '1d_return' in sector_price_df.columns:
                    price_df['1d_return'] = sector_price_df['1d_return']
            
            return price_df.sort_index() if not price_df.empty else price_df
    
    return FixedLegacyFeaturesFacade

def test_fixed_facade():
    """修正版ファサードのテスト"""
    print("\n=== 修正版ファサードテスト ===")
    
    try:
        # 修正版ファサードを作成
        FixedFacade = create_fixed_legacy_facade()
        facade = FixedFacade(use_new_system=True, validate_consistency=False)
        
        # 新システムで計算
        result = facade.calculate_features(
            new_sector_price=sector_price_df,
            new_sector_list=sector_list_df,
            stock_dfs_dict=stock_dfs_dict,
            adopts_features_indices=False,  # 簡易テストのため無効化
            adopts_features_price=True,
            adopt_1d_return=True,
            mom_duration=[5, 21],
            vola_duration=[5, 21],
            adopt_size_factor=False,
            adopt_eps_factor=False,
            adopt_sector_categorical=True,
            add_rank=True
        )
        
        print(f"✓ 修正版ファサード成功")
        print(f"  出力形状: {result.shape}")
        print(f"  出力カラム: {list(result.columns)}")
        
        return result
        
    except Exception as e:
        print(f"✗ 修正版ファサードエラー: {e}")
        import traceback
        traceback.print_exc()
        return None

# 修正版ファサードをテスト
fixed_facade_result = test_fixed_facade()

# %% [markdown]
# ## 7. 総合検証レポート

# %%
def generate_validation_report():
    """総合検証レポートを生成"""
    print("\n" + "="*60)
    print("データ処理システム検証レポート")
    print("="*60)
    
    # 各コンポーネントの状態
    components_status = {
        "サンプルデータ作成": sector_price_df is not None,
        "データ前処理": processed_sector_df is not None,
        "モメンタム変換器": momentum_result is not None,
        "ボラティリティ変換器": volatility_result is not None,
        "ランキング変換器": ranking_result is not None,
        "セクターカテゴリ変換器": sector_cat_result is not None,
        "統合パイプライン": integrated_result is not None,
        "修正版ファサード": fixed_facade_result is not None,
    }
    
    print("\n### コンポーネント動作状況:")
    for component, status in components_status.items():
        status_mark = "✓" if status else "✗"
        print(f"  {status_mark} {component}")
    
    # 成功率
    success_rate = sum(components_status.values()) / len(components_status) * 100
    print(f"\n### 全体成功率: {success_rate:.1f}%")
    
    # データ品質
    if integrated_result is not None:
        print(f"\n### データ品質（統合パイプライン結果）:")
        print(f"  データ形状: {integrated_result.shape}")
        print(f"  カラム数: {len(integrated_result.columns)}")
        
        # 欠損値率
        missing_rates = integrated_result.isnull().mean()
        high_missing = missing_rates[missing_rates > 0.1]
        if not high_missing.empty:
            print(f"  高欠損値カラム（>10%）: {len(high_missing)}個")
        else:
            print(f"  ✓ 全カラムの欠損値率が10%以下")
    
    # 比較結果
    if comparison_results:
        high_corr_count = sum(1 for r in comparison_results.values() 
                             if not pd.isna(r['correlation']) and r['correlation'] > 0.95)
        total_compared = len(comparison_results)
        print(f"\n### 新旧システム比較:")
        print(f"  比較カラム数: {total_compared}")
        print(f"  高相関カラム数（>0.95）: {high_corr_count}")
        if total_compared > 0:
            print(f"  一致率: {high_corr_count/total_compared*100:.1f}%")
    
    # 推奨事項
    print(f"\n### 推奨事項:")
    if success_rate >= 80:
        print("  ✓ システムは概ね正常に動作しています")
        print("  → 本格的な移行準備を開始できます")
    elif success_rate >= 60:
        print("  ⚠ 一部のコンポーネントに問題があります")
        print("  → 失敗したコンポーネントの修正が必要です")
    else:
        print("  ✗ 多くのコンポーネントに問題があります")
        print("  → 根本的な見直しが必要です")
    
    if comparison_results and len(comparison_results) > 0:
        avg_corr = np.mean([r['correlation'] for r in comparison_results.values() 
                           if not pd.isna(r['correlation'])])
        if avg_corr >= 0.95:
            print("  ✓ 新旧システムの一致率が高く、移行リスクは低いです")
        elif avg_corr >= 0.80:
            print("  ⚠ 新旧システムに一部差異があります。詳細検証が必要です")
        else:
            print("  ✗ 新旧システムに大きな差異があります。修正が必要です")
    
    print("\n" + "="*60)

# 総合検証レポートを生成
generate_validation_report()

# %% [markdown]
# ## 8. 修正提案と次のステップ

# %%
print("\n" + "="*60)
print("修正提案と次のステップ")
print("="*60)

print("""
### 修正提案:

1. **インデックス処理の統一**:
   - ランキング計算時の`Date`レベルアクセスを修正
   - マルチインデックスの一貫した処理方法を確立

2. **ファサードの改善**:
   - 既存システムとの型互換性を向上
   - エラーハンドリングの強化
   - フォールバック機構の実装

3. **データ検証の強化**:
   - 入力データの契約検証を強化
   - 中間結果の整合性チェック追加

### 次のステップ:

1. **段階的ロールアウト**:
   - まず価格系特徴量から新システムに移行
   - インデックス系特徴量は既存システムを継続使用
   - 徐々に新システムの機能を拡張

2. **パフォーマンス最適化**:
   - 大規模データでの性能テスト
   - メモリ使用量の最適化
   - 並列処理の導入検討

3. **運用監視**:
   - 新旧システムの継続的比較
   - 異常値検出の自動化
   - アラート機能の実装

4. **ドキュメント整備**:
   - 移行ガイドの作成
   - トラブルシューティングガイド
   - ベストプラクティスの文書化
""")

print("="*60)