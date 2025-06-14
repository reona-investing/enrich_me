# calculation/sector_index/calculators のクラス仕様書

## index_calculator.py

### class SectorIndexCalculator
セクター別の株価データからインデックス値を算出するユーティリティ。
- aggregate: セクターインデックスの基礎データを集約する。
- _calculate_ohlc: 集約後のデータから OHLC を求める補助メソッド。

## marketcap_calculator.py

### class MarketCapCalculator
株価データに発行済み株式数を付与し時価総額を計算する補助クラス。
- merge_stock_price_and_shares: 株価データに発行済み株式数情報を結合する。
- _calc_shares_at_end_period: 決算期末時点の発行済み株式数を抽出する。
- _append_next_period_start_date: 次会計期間の開始日を営業日に丸める。
- _find_next_business_day: 与えられた日付以降で最初の営業日を返す。
- _merge_with_stock_price: 株価データと株式数情報をマージする。
- calc_adjustment_factor: 株式分割などの調整係数を計算する。
- _extract_rows_to_adjust: 分割調整が必要な行を抽出する。
- _calc_shares_rate: 株式数変化率を計算する。
- _correct_shares_rate_for_non_adjustment: 調整不要なケースを考慮して ``SharesRate`` を補正する。
- _merge_shares_rate: 株数変化率を株価データにマージする。
- _handle_special_cases: 特殊な銘柄の ``SharesRate`` を補正する。
- _calc_cumulative_shares_rate: 変化率を累積して連続的な補正率を算出する。
- adjust_shares: 株式数を ``CumulativeSharesRate`` で補正する。
- calc_marketcap: OHLC 各値の時価総額を計算する。
- calc_correction_value: 指数算出用の補正値を計算する。

