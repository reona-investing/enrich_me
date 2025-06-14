# preprocessing/pipeline のクラス仕様書

## pipeline.py

### class PreprocessingPipeline
- __init__: sklearn.pipeline.Pipeline をラップするクラス

Parameters
----------
steps : List[tuple[str, Transformer]]
    sklearnと同様の形式で処理ステップを渡す
- fit: 
- transform: 
- fit_transform: 
- _validate_input: 
- _check_is_fitted: 
- get_pipeline: 内部のPipelineインスタンスを取得
- get_feature_names_out: 最終ステップの出力カラム名を取得できる場合は取得

