import pandas as pd
import numpy as np
from datetime import datetime
from utils.paths import Paths
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ========== テストデータの作成 ==========
def create_test_data():
    # 日付の作成
    dates = pd.date_range(start='2023-01-01', end='2023-12-31')
    
    # セクターの定義
    sectors = ['Tech', 'Healthcare', 'Finance', 'Energy', 'Consumer']
    
    # 特徴量とターゲットのデータを作成
    data = []
    for date in dates:
        for sector in sectors:
            # ランダムな特徴量を生成
            feature_1 = np.random.normal(0, 1)
            feature_2 = np.random.normal(0, 1)
            feature_3 = np.random.normal(0, 1)
            feature_4 = np.random.normal(0, 1)
            
            # 特定の関係性を持ったターゲットを生成
            noise = np.random.normal(0, 0.5)
            target = 0.5 * feature_1 - 0.3 * feature_2 + 0.7 * feature_3 + noise
            
            # セクター特有の効果を追加
            if sector == 'Tech':
                target += 0.2
            elif sector == 'Finance':
                target -= 0.15
            
            data.append([date, sector, feature_1, feature_2, feature_3, feature_4, target])
    
    # データフレームに変換
    df = pd.DataFrame(data, columns=['Date', 'Sector', 'Feature1', 'Feature2', 'Feature3', 'Feature4', 'Target'])
    
    # インデックスを設定
    df = df.set_index(['Date', 'Sector'])
    
    # 特徴量とターゲットを分離
    features_df = df.drop(columns=['Target'])
    target_df = df[['Target']]
    
    return features_df, target_df

# ========== 旧モデルのテスト ==========
def test_old_model(features_df, target_df):
    print("=== 旧モデル（models）のテスト ===")
    
    from models import LassoModel as OldLassoModel, LgbmModel as OldLgbmModel
    
    # データを訓練用とテスト用に分割
    train_dates = pd.date_range(start='2023-01-01', end='2023-10-31')
    test_dates = pd.date_range(start='2023-11-01', end='2023-12-31')
    
    features_train = features_df.loc[features_df.index.get_level_values('Date').isin(train_dates)]
    target_train = target_df.loc[target_df.index.get_level_values('Date').isin(train_dates)]
    features_test = features_df.loc[features_df.index.get_level_values('Date').isin(test_dates)]
    target_test = target_df.loc[target_df.index.get_level_values('Date').isin(test_dates)]
    
    # Lassoモデルのテスト
    print("Lassoモデルの訓練...")
    lasso_model = OldLassoModel()
    lasso_output = lasso_model.train(target_train, features_train, max_features=3, min_features=1)
    
    print("Lassoモデルの予測...")
    lasso_predictions = lasso_model.predict(target_test, features_test, lasso_output.models, lasso_output.scalers)
    
    # LightGBMモデルのテスト
    print("LightGBMモデルの訓練...")
    lgbm_model = OldLgbmModel()
    lgbm_output = lgbm_model.train(target_train, features_train)
    
    print("LightGBMモデルの予測...")
    lgbm_predictions = lgbm_model.predict(target_test, features_test, lgbm_output.models)
    
    return lasso_predictions, lgbm_predictions

# ========== 新モデルのテスト ==========
def test_new_model(features_df, target_df):
    print("=== 新モデル（models2）のテスト ===")
    
    from models2 import LassoModel, LgbmModel, ModelFactory
    
    # データを訓練用とテスト用に分割
    train_dates = pd.date_range(start='2023-01-01', end='2023-10-31')
    test_dates = pd.date_range(start='2023-11-01', end='2023-12-31')
    
    features_train = features_df.loc[features_df.index.get_level_values('Date').isin(train_dates)]
    target_train = target_df.loc[target_df.index.get_level_values('Date').isin(train_dates)]
    features_test = features_df.loc[features_df.index.get_level_values('Date').isin(test_dates)]
    target_test = target_df.loc[target_df.index.get_level_values('Date').isin(test_dates)]
    
    # Lassoモデルのテスト
    print("Lassoモデルの訓練...")
    lasso_model = LassoModel()
    lasso_model.train(features_train, target_train, max_features=3, min_features=1)
    
    print("Lassoモデルの予測...")
    lasso_predictions = pd.DataFrame({'Pred': lasso_model.predict(features_test)}, index=target_test.index)
    
    # LightGBMモデルのテスト
    print("LightGBMモデルの訓練...")
    lgbm_model = LgbmModel()
    lgbm_model.train(features_train, target_train)
    
    print("LightGBMモデルの予測...")
    lgbm_predictions = pd.DataFrame({'Pred': lgbm_model.predict(features_test)}, index=target_test.index)
    
    # モデルコンテナを使った例
    print("モデルコンテナのテスト...")
    sectors = features_df.index.get_level_values('Sector').unique()
    container = ModelFactory.create_lasso_container(sectors)
    
    # セクターごとに学習
    container.train(features_train, target_train, max_features=3, min_features=1)
    
    # 特徴量重要度の確認
    for sector, importances in container.get_feature_importances().items():
        print(f"セクター '{sector}' の特徴量重要度トップ3:")
        print(importances.head(3))
    
    return lasso_predictions, lgbm_predictions, container

# ========== 予測結果の比較 ==========
def compare_predictions(old_lasso_predictions, old_lgbm_predictions, 
                       new_lasso_predictions, new_lgbm_predictions, target_test):
    print("\n=== 予測結果の比較 ===")
    
    # 旧モデルの予測をマージ
    old_predictions = pd.DataFrame({
        'Target': target_test['Target'],
        'Old_Lasso_Pred': old_lasso_predictions['Pred'],
        'Old_LGBM_Pred': old_lgbm_predictions['Pred']
    })
    
    # 新モデルの予測をマージ
    new_predictions = pd.DataFrame({
        'Target': target_test['Target'],
        'New_Lasso_Pred': new_lasso_predictions['Pred'],
        'New_LGBM_Pred': new_lgbm_predictions['Pred']
    })
    
    # 全予測結果をマージ
    all_predictions = pd.concat([old_predictions, new_predictions.drop(columns=['Target'])], axis=1)
    
    # 相関係数を計算
    correlation_matrix = all_predictions.corr()
    print("相関係数:")
    print(correlation_matrix)
    
    # 統計情報の比較
    stats = pd.DataFrame({
        'Old_Lasso': old_lasso_predictions['Pred'].describe(),
        'Old_LGBM': old_lgbm_predictions['Pred'].describe(),
        'New_Lasso': new_lasso_predictions['Pred'].describe(),
        'New_LGBM': new_lgbm_predictions['Pred'].describe(),
        'Target': target_test['Target'].describe()
    })
    print("\n統計情報の比較:")
    print(stats)
    
    # モデル間の予測差の分布
    diff_lasso = old_lasso_predictions['Pred'] - new_lasso_predictions['Pred']
    diff_lgbm = old_lgbm_predictions['Pred'] - new_lgbm_predictions['Pred']
    
    print("\n旧モデルと新モデルの予測差の統計:")
    print("Lasso予測差の統計:")
    print(diff_lasso.describe())
    print("LGBM予測差の統計:")
    print(diff_lgbm.describe())
    
    # 散布図で視覚化
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.scatter(old_lasso_predictions['Pred'], target_test['Target'], alpha=0.5, label='Old Lasso')
    plt.scatter(new_lasso_predictions['Pred'], target_test['Target'], alpha=0.5, label='New Lasso')
    plt.title('Lasso Predictions vs Target')
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.scatter(old_lgbm_predictions['Pred'], target_test['Target'], alpha=0.5, label='Old LGBM')
    plt.scatter(new_lgbm_predictions['Pred'], target_test['Target'], alpha=0.5, label='New LGBM')
    plt.title('LGBM Predictions vs Target')
    plt.xlabel('Predicted')
    plt.ylabel('Target')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.hist(diff_lasso, bins=20, alpha=0.7)
    plt.title('Distribution of Lasso Prediction Differences (Old - New)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    
    plt.subplot(2, 2, 4)
    plt.hist(diff_lgbm, bins=20, alpha=0.7)
    plt.title('Distribution of LGBM Prediction Differences (Old - New)')
    plt.xlabel('Difference')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{Paths.SUMMARY_REPORTS_FOLDER}/model_comparison.png')
    plt.show()
    
    return all_predictions

# ========== メイン関数 ==========
def main():
    # テストデータの作成
    features_df, target_df = create_test_data()
    print(f"作成されたデータ: {features_df.shape[0]}行 x {features_df.shape[1]}列（特徴量）")
    print(f"データの日付範囲: {features_df.index.get_level_values('Date').min()} から {features_df.index.get_level_values('Date').max()}")
    print(f"セクターの種類: {features_df.index.get_level_values('Sector').unique()}")
    
    # 旧モデルのテスト
    old_lasso_predictions, old_lgbm_predictions = test_old_model(features_df, target_df)
    
    # 新モデルのテスト
    new_lasso_predictions, new_lgbm_predictions, container = test_new_model(features_df, target_df)
    
    # データの分割
    test_dates = pd.date_range(start='2023-11-01', end='2023-12-31')
    target_test = target_df.loc[target_df.index.get_level_values('Date').isin(test_dates)]
    
    # 予測結果の比較
    all_predictions = compare_predictions(
        old_lasso_predictions, old_lgbm_predictions,
        new_lasso_predictions, new_lgbm_predictions, 
        target_test
    )
    
    print("テスト完了。結果は上記のとおりです。")
    return all_predictions

if __name__ == "__main__":
    main()