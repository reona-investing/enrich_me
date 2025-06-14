# acquisition/features_updater/facades のクラス仕様書

## features_update_facade.py

### class FeaturesUpdateFacade
特徴量データ更新のファサードクラス（リファクタリング版）
- __init__: Args:
    max_concurrent_tasks (int): 同時実行タスク数の上限
- _load_features_config: 設定ファイルから特徴量設定を読み込み
- _display_progress: 進捗表示
- _summarize_results: 結果の集計
- _display_final_summary: 最終結果の表示
- get_features_status: 全特徴量の状況を取得

