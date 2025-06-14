# models/machine_learning/ml_dataset/components のクラス仕様書

## base_data_component.py

### class BaseDataComponent
データコンポーネントの抽象基底クラス
- __init__: 
- _load_all_data: 全データをロード
- _load_file: 単一ファイルをロード
- save_instance: インスタンスを保存
- _save_file: 単一ファイルを保存
- getter: データクラスとして返却（サブクラスで実装）

## ml_objects.py

### class MLObjects
機械学習オブジェクトの管理
- archive_ml_objects: 機械学習のモデルとスケーラーを格納
- getter: データクラスとして返却

``_model`` や ``_scaler`` が存在しない場合は ``None`` を返す。
予測のみを行う際に ``archive_ml_objects`` が呼ばれていない状況でも
エラーとならないようにしている。

## post_processing_data.py

### class PostProcessingData
後処理データの管理
- archive_raw_target: 生の目的変数を格納
- archive_order_price: 個別銘柄の発注価格を格納
- archive_pred_result: 予測結果を格納
- getter_stock_selection: 株式選択用データを返却
- getter_evaluation: 評価用データを返却
- getter: デフォルトのgetter（評価用データを返却）

## train_test_data.py

### class TrainTestData
訓練・テストデータの管理と前処理を担当
- archive: データの前処理と分割を実行
- _prepare_target_data: 目的変数データの前処理
- _prepare_features_data: 特徴量データの前処理
- _split_data: データを学習・テストに分割
- _remove_outliers: 外れ値除去
- _append_next_business_day_row: 次の営業日の行を追加
- _shift_features: 特徴量を1日シフト
- _align_index: 特徴量のインデックスを目的変数と揃える
- _narrow_period: 指定期間でデータを絞り込み
- _filter_outliers_from_datasets: 外れ値除去の実行
- _filter_outliers_by_group: グループ単位での外れ値除去
- getter: データクラスとして返却

