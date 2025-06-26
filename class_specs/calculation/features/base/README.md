# calculation/features/base のクラス仕様書

## base_features.py

### class BaseFeatures
特徴量計算の基底クラス
- __init__: 共通の初期化処理
- calculate_features: 特徴量を計算する抽象メソッド
実装クラスはこのメソッド内でself.features_dfを更新する必要がある
- apply_preprocessing: 前処理パイプラインを適用し、self.features_dfを更新

Args:
    pipeline: 前処理パイプライン
    
Returns:
    前処理後の特徴量データフレーム
- get_features: 計算済み特徴量の安全なコピーを取得

Returns:
    特徴量データフレームのコピー
    
Raises:
    ValueError: 特徴量が計算されていない場合
- has_features: 特徴量が計算済みかどうかを確認

Returns:
    特徴量が存在するかどうか
- clear_features: 特徴量データをクリア

