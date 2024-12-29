#%% モジュールのインポート
#パスを通す
if __name__ == '__main__':
    from pathlib import Path
    import sys
    PROJECT_FOLDER = str(Path(__file__).parents[2])
    ORIGINAL_MODULES = PROJECT_FOLDER + '/modules'
    sys.path.append(ORIGINAL_MODULES)

# モジュールのインポート
from datetime import datetime

import paths #パス一覧
from jquants_api_operations import run_jquants_api_operations
import sector_index_calculator
from models import MLDataset
import asyncio

#%% メイン関数
async def main(ML_DATASET_PATH:str, NEW_SECTOR_LIST_CSV:str, NEW_SECTOR_PRICE_PKLGZ:str,
         universe_filter:str, trading_sector_num:int, candidate_sector_num:int,
         train_start_day:datetime, train_end_day:datetime,
         test_start_day:datetime, test_end_day:datetime,
         top_slope:float = 1.0, should_learn:bool = True):
    '''
    モデルの実装
    ML_DATASET_PATH: 学習済みモデル、スケーラー、予測結果等を格納したデータセット
    NEW_SECTOR_LIST_CSV: 銘柄と業種の対応リスト
    NEW_SECTOR_PRICE_PKLGZ: 業種別の株価インデックス
    universe_filter: ユニバースを絞るフィルタ
    trading_sector_num: 上位・下位何業種を取引対象とするか
    candidate_sector_num: 取引できない業種がある場合、上位・下位何業種を取引対象候補とするか
    train_start_day: 学習期間の開始日
    train_end_day: 学習期間の終了日
    test_start_day: テスト期間の開始日
    test_end_day: テスト期間の終了日
    top_slope: トップ予想の業種にどれほどの傾斜をかけるか
    should_learn: 学習するか否か
    '''
    # ml_datasetは必ず生成するので、最初に生成してしまう。
    ml_dataset = MLDataset(ML_DATASET_PATH)
    list_df, fin_df, price_df = run_jquants_api_operations(filter = universe_filter)
    stock_dfs_dict = {'stock_list': list_df, 'stock_fin': fin_df, 'stock_price': price_df}

    new_sector_price_df, order_price_df = sector_index_calculator.calc_new_sector_price(stock_dfs_dict, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ)

    '''リターン予測→ml_datasetの作成'''
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import pandas as pd
    from sklearn.preprocessing import OneHotEncoder
    import numpy as np

    # One-Hot Encodingを適用
    encoder = OneHotEncoder(sparse_output=False) 
    ml_dataset.features_train_df = ml_dataset.features_train_df.reset_index(drop=False)
    ml_dataset.features_test_df = ml_dataset.features_test_df.reset_index(drop=False)

    encoded_train = encoder.fit_transform(ml_dataset.features_train_df[['Sector']])
    encoded_test = encoder.transform(ml_dataset.features_test_df[['Sector']])

    # One-Hot Encodingされたカテゴリ変数を元のデータに追加
    encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(['Sector']))
    encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(['Sector']))

    features_train = pd.concat([ml_dataset.features_train_df.reset_index(drop=True).drop('Sector', axis=1), encoded_train_df], axis=1)
    features_test = pd.concat([ml_dataset.features_test_df.reset_index(drop=True).drop('Sector', axis=1), encoded_test_df], axis=1)

    # データ型をfloat32に変換
    features_train = features_train.drop('Date', axis=1).astype(np.float32)
    features_test = features_test.drop('Date', axis=1).astype(np.float32)
    target_train = ml_dataset.target_train_df.astype(np.float32)
    target_test = ml_dataset.target_test_df.astype(np.float32)

    # 'Date'と'Sector'でグループ化し、シーケンスごとにデータを抽出
    def create_sequences(df, sequence_length):
        sequences = []
        for _, group in df.groupby('Date'):
            if len(group) == sequence_length:
                sequences.append(group.drop(['Date', 'Sector'], axis=1).values)
        return np.array(sequences)

    # テンソルに変換
    features_train_tensor = torch.tensor(features_train.values, dtype=torch.float32)
    features_test_tensor = torch.tensor(features_test.values, dtype=torch.float32)
    target_train_tensor = torch.tensor(target_train.values, dtype=torch.float32)
    target_test_tensor = torch.tensor(target_test.values, dtype=torch.float32)

    # GPUが利用可能か確認
    if torch.cuda.is_available():
        print('GPUが利用可能です。')
        features_train_tensor = features_train_tensor.to('cuda')
        target_train_tensor = target_train_tensor.to('cuda')
        features_test_tensor = features_test_tensor.to('cuda')
        target_test_tensor = target_test_tensor.to('cuda')

    # TensorDatasetを作成
    train_dataset = TensorDataset(features_train_tensor, target_train_tensor)
    test_dataset = TensorDataset(features_test_tensor, target_test_tensor)

    # DataLoaderを作成
    batch_size = 48
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # シンプルなRNNモデルの定義
    class StockPredictorRNN(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers):
            super(StockPredictorRNN, self).__init__()
            self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 1)
        
        def forward(self, x):
            h0 = torch.zeros(num_layers, x.size(0), hidden_dim).to(x.device)
            out, _ = self.rnn(x, h0)
            out = self.fc(out[:, -1, :])
            return out

    # モデル、損失関数、最適化手法の定義
    input_dim = features_train_tensor.shape[1]  # 特徴量の次元数
    hidden_dim = 64
    num_layers = 2
    model = StockPredictorRNN(input_dim, hidden_dim, num_layers)
    if torch.cuda.is_available():
        model = model.to('cuda')
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # トレーニングループ
    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            print(batch_targets.shape)
            print(batch_features.shape)
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # テストデータでの予測
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch_features, _ in test_loader:
            outputs = model(batch_features)
            predictions.append(outputs)
        predictions = torch.cat(predictions, dim=0)

    print("Predictions on test data:", predictions)

    return predictions


#%% パラメータ類
if __name__ == '__main__':
    '''パス類'''
    NEW_SECTOR_LIST_CSV = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_list.csv' #別でファイルを作っておく
    NEW_SECTOR_PRICE_PKLGZ = f'{paths.SECTOR_REDEFINITIONS_FOLDER}/New48sectors_price.pkl.gz' #出力のみなのでファイルがなくてもOK
    ML_DATASET_PATH = f'{paths.ML_DATASETS_FOLDER}/New48sectors.pkl.gz'
    '''ユニバースを絞るフィルタ'''
    universe_filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))" #現行のTOPIX500
    '''上位・下位何業種を取引対象とするか？'''
    trading_sector_num = 3
    candidate_sector_num = 5
    '''トップ予想の業種にどれほどの傾斜をかけるか'''
    top_slope = 1.5
    '''学習期間'''
    train_start_day = datetime(2014, 1, 1)
    train_end_day = datetime(2021, 12, 31)
    test_start_day = datetime(2014, 1, 1)
    test_end_day = datetime(2099, 12, 31) #ずっと先の未来を指定
    '''学習するか否か'''
    should_learn = True

#%% 実行
if __name__ == '__main__':
    predictions = asyncio.run(main(ML_DATASET_PATH, NEW_SECTOR_LIST_CSV, NEW_SECTOR_PRICE_PKLGZ,
                            universe_filter, trading_sector_num, candidate_sector_num,
                            train_start_day, train_end_day, test_start_day, test_end_day,
                            top_slope, should_learn))
    
#%% 出力のみなのでファイルがなくてもOK
if __name__ == '__main__':
    import pandas as pd
    from IPython.display import display
    from models import MLDataset
    ml_dataset = MLDataset(ML_DATASET_PATH)
    pred_df = pd.DataFrame(predictions, index=ml_dataset.target_test_df.index, columns=['Pred'])
    pred_df = pd.concat(ml_dataset.raw_target_df, pred_df, axis=1)
    display(pred_df)
    pred_df.to_csv('deep_test.csv')

