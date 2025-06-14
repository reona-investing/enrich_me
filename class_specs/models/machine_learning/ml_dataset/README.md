# models/machine_learning/ml_dataset のクラス仕様書

## ml_datasets.py

### class MLDatasets
複数のSingleMLDatasetを統合管理するクラス
- __init__: 
- append_model: SingleMLDatasetを追加
- remove_model: 指定された名前のモデルを削除
- replace_model: 既存のモデルを差し替え（同じ名前のモデルが存在する場合）
- get_model: 指定された名前のモデルを取得
- get_model_names: 登録されているモデル名の一覧を取得
- _merge_dfs: 各モデルからDataFrameを取得して結合するヘルパー
- get_pred_result: 各SingleMLDatasetの予測結果dfをconcatして日付順に並べたものを出力

Args:
    model_names (Optional[List[str]]): 対象とするモデル名のリスト。
                                      Noneの場合は全モデルを対象とする。

Returns:
    pd.DataFrame: 統合された予測結果データフレーム
- get_raw_target: raw_target_dfをマージして返却
- get_order_price: order_price_dfをマージして返却
- save_all: 全てのモデルを保存
- __len__: 登録されているモデル数を返す
- __contains__: 指定されたモデル名が登録されているかチェック
- __iter__: イテレータ（モデル名でイテレート）
- items: (モデル名, SingleMLDataset)のペアを返す

## single_ml_dataset.py

### class SingleMLDataset
単体機械学習データセットの統合管理
- __init__: 
- get_name: オブジェクトの名称を取得
- save: 全体を保存
- archive_train_test_data: TrainTestData の archive メソッドを実行
- archive_ml_objects: MLObjects の archive メソッドを実行
- archive_post_processing_data: PostProcessingData の archive メソッドを実行
- archive_raw_target: raw_target_dfをアーカイブ
- archive_order_price: order_price_dfをアーカイブ
- archive_pred_result: pred_result_dfをアーカイブ
- train_test_materials: 
- ml_object_materials: 
- evaluation_materials: 
- stock_selection_materials: 
- copy_from_other_dataset: 他のデータセットからすべてのインスタンス変数をコピー

