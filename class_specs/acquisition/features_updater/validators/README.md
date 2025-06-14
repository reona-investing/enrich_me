# acquisition/features_updater/validators のクラス仕様書

## data_validator.py

### class FeatureDataValidator
特徴量データの整合性検証に特化したクラス
- __init__: 
- validate_data_integrity: データの整合性を検証

Args:
    df (pd.DataFrame): 検証対象のデータ
    
Returns:
    tuple[bool, list[str]]: (検証結果, エラーメッセージリスト)
- validate_date_continuity: 日付の連続性を検証

