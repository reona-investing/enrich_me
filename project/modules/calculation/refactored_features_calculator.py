import pandas as pd
import logging
import re
from typing import Literal, List, Optional
from utils.paths import Paths
from utils.metadata import FeatureMetadata
from calculation import SectorIndexCalculator
from utils import yaml_utils

logging.basicConfig(level=logging.ERROR)


class IndexFeatureCalculator:
    """
    インデックス系の特徴量（通貨・債券・コモディティなど）を計算するクラス。

    主な処理フロー:
      1. 各種メタデータ (FeatureMetadata) から parquetファイルを読み込み、
         リターン (pct_change や diff) を算出して self.features_df に結合。
      2. finalize() メソッドで通貨の相対強度と債券の差分を処理し、
         欠損値を最新値で埋め (ffill) た最終的な特徴量を返す。

    注意事項:
      - 通貨ペアはISO 4217に準拠した6文字 (例: "USDJPY") を想定し、
        先頭3文字がベース通貨・後ろ3文字がクオート通貨。
      - 債券は "bond" という文字列が含まれ、末尾に年数(数字)が付く形を想定。
      - finalize() における欠損値補完は ffill() 固定としている。
      - 通貨ペアの各列 (例: "USDJPY_1d_return") は最終的に削除し、
        各通貨単独 (例: "USD_1d_return") の相対強度列だけを残す。

    Attributes:
        features_df (pd.DataFrame): 計算途中および最終的な特徴量を格納したDataFrame。
        metadata_list (List[FeatureMetadata]): 採用したメタデータを格納するリスト。
        currencies_type (Literal['relative', 'raw']): 通貨を相対強度 ("relative") か生データ ("raw") で扱うか。
    """

    def __init__(self, currencies_type: Literal['relative', 'raw'] = 'relative'):
        """
        コンストラクタ

        Args:
            currencies_type (Literal['relative', 'raw']):
                通貨ペアのリターンを「相対強度 (relative)」に変換するか、
                あるいはそのまま (raw) で保持するかを指定する。
        """
        self.features_df = pd.DataFrame(columns=['Date'])
        self.metadata_list = []
        self.currencies_type = currencies_type

    def calculate_return(self, feature_metadata: FeatureMetadata, days: int = 1):
        """
        指定された特徴量メタデータに基づいてリターンを計算し、features_dfに結合する。

        債券の場合は差分(diff)を算出し、通貨・コモディティなどそれ以外の場合は
        pct_change でリターンを算出する。

        Args:
            feature_metadata (FeatureMetadata): 特徴量のメタデータ。
            days (int): 何日リターンを算出するか。デフォルトは1日。
        """
        if not feature_metadata.is_adopted:
            # 採用フラグがFalseの場合は処理スキップ
            return

        try:
            raw_df = pd.read_parquet(feature_metadata.parquet_path)
            feature = pd.DataFrame()
            feature["Date"] = raw_df["Date"]

            if feature_metadata.group == "bond":
                # 債券 -> diff
                col_name = f"{feature_metadata.name}_{days}d_return"
                feature[col_name] = raw_df["Close"].diff(days)
            else:
                # 通貨・コモディティなど -> pct_change
                col_name = f"{feature_metadata.name}_{days}d_return"
                feature[col_name] = raw_df["Close"].pct_change(days)

            self.features_df = pd.merge(
                self.features_df, feature, how='outer', on='Date'
            ).sort_values(by='Date')

            self.metadata_list.append(feature_metadata)

        except FileNotFoundError as e:
            logging.error(f"ファイルが見つかりません: {feature_metadata.parquet_path} | エラー内容: {e}")
        except Exception as e:
            logging.error(f"{feature_metadata.name} のリターン算出中にエラーが発生しました: {e}")

    def finalize(self) -> pd.DataFrame:
        """
        通貨の相対強度処理と債券の差分処理を行い、欠損値を最新値で埋めた最終的な特徴量DataFrameを返す。

        Returns:
            pd.DataFrame: "Date" をインデックスとした最終的なインデックス特徴量DataFrame。
        """
        # 時系列順に並べ、最新値で ffill() して欠損補完
        features_df = self.features_df.sort_values(by='Date').set_index('Date')
        features_df = features_df.ffill()

        # 通貨相対強度 (relative指定の場合)
        if self.currencies_type == 'relative':
            features_df = self._relativizate_currency(features_df)

        # 債券差分を作成
        features_df = self._process_bond(features_df)

        return features_df

    # --------------------------------------------------------------------------
    # 以下、_relativizate_currency と _process_bond を
    # ヘルパー関数に分割した実装例。
    # --------------------------------------------------------------------------

    def _relativizate_currency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        通貨ペア列 ( {6文字通貨ペア}_{suffix} ) を相対強度に変換し、
        元の通貨ペア列を削除した上で、各通貨単独の列のみ残す。

        Args:
            df (pd.DataFrame): リターン計算済みの特徴量を格納したDataFrame。

        Returns:
            pd.DataFrame: 通貨ペア列を削除し、各通貨の相対強度列のみを追加したDataFrame。
        """
        # メタデータ上で通貨グループに属するものを抽出
        currency_pairs = [m.name for m in self.metadata_list if m.group == 'currency' and m.is_adopted]

        # 通貨関連の列を収集 (例: "USDJPY_1d_return", "EURUSD_1d_return" ...)
        currency_cols = [
            col for col in df.columns
            for pair in currency_pairs
            if col.startswith(pair + '_')
        ]

        # suffix一覧を取得 (例: "1d_return", "5d_return", ...)
        suffixes = {col.split('_', 1)[1] for col in currency_cols}

        for suffix in suffixes:
            # suffix単位で通貨ペア列を集約
            cols_for_suffix = [col for col in currency_cols if col.endswith(suffix)]
            df = self._process_currency_suffix(df, cols_for_suffix, suffix)

        return df

    def _process_currency_suffix(self, df: pd.DataFrame, cols_for_suffix: List[str], suffix: str) -> pd.DataFrame:
        """
        特定のsuffixに該当する通貨ペア列を相対強度に再集計し、
        最後に通貨ペア列をまとめて削除して各通貨列のみ残す。

        例: "USDJPY_1d_return" → "USD_1d_return" (加算), "JPY_1d_return" (減算)

        Args:
            df (pd.DataFrame): リターン計算済みの特徴量DataFrame。
            cols_for_suffix (List[str]): 対象となる通貨ペア列リスト (例: ["USDJPY_1d_return", ...]).
            suffix (str): 変換対象のsuffix (例: "1d_return").

        Returns:
            pd.DataFrame: 通貨ペア列を削除し、各通貨の相対強度を残したDataFrame。
        """
        # 通貨コードごとの加算回数をカウントするための辞書
        currency_counts = {}

        # 1) 全ペアを確認し、各通貨コードを初期化
        for col in cols_for_suffix:
            pair = col.split('_')[0]  # 例: "USDJPY"
            base = pair[:3]          # 例: "USD"
            quote = pair[3:]         # 例: "JPY"
            currency_counts.setdefault(base, 0)
            currency_counts.setdefault(quote, 0)

        # 2) 通貨コードごとの列を新設し、0.0で初期化
        for cur in currency_counts.keys():
            new_col = f"{cur}_{suffix}"  # 例: "USD_1d_return"
            if new_col not in df.columns:
                df[new_col] = 0.0

        # 3) 各ペア列を読み込み、base通貨に加算、quote通貨に減算
        for col in cols_for_suffix:
            pair = col.split('_')[0]
            base = pair[:3]
            quote = pair[3:]

            # base通貨：加算
            df[f"{base}_{suffix}"] += df[col]
            currency_counts[base] += 1

            # quote通貨：減算
            df[f"{quote}_{suffix}"] -= df[col]
            currency_counts[quote] += 1

        # 4) 各通貨ごとに加算回数で割って平均を取る
        for cur, cnt in currency_counts.items():
            if cnt > 0:
                df[f"{cur}_{suffix}"] /= cnt

        # 5) 最後に、元の通貨ペア列を一括で削除する
        df = df.drop(columns=cols_for_suffix)

        return df

    def _process_bond(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        債券特徴量 (XXXbond{年数}_{suffix}) について、
        すべての年数ペアの差分(長期 - 短期)を計算して列を追加。

        Args:
            df (pd.DataFrame): リターン計算済みの特徴量DataFrame。

        Returns:
            pd.DataFrame: 債券差分列を追加したDataFrame。
        """
        # メタデータからbondグループのnameを抽出
        bond_names = [m.name for m in self.metadata_list if m.group == 'bond' and m.is_adopted]
        # DataFrame上でbond列だけを抽出
        bond_cols = [
            col for col in df.columns
            for bn in bond_names
            if col.startswith(bn + '_')
        ]

        # suffixごとにまとめて処理
        suffixes = {col.split('_', 1)[1] for col in bond_cols if '_' in col}
        for sfx in suffixes:
            cols_for_suffix = [c for c in bond_cols if c.endswith(sfx)]
            df = self._process_bond_for_suffix(df, cols_for_suffix, sfx)

        return df

    def _process_bond_for_suffix(self, df: pd.DataFrame,
                                 bond_cols_for_suffix: List[str],
                                 suffix: str) -> pd.DataFrame:
        """
        特定のsuffixを持つ債券列(XXXbond{年数}_{suffix})間の
        全ペア差分(長期 - 短期)を計算して列を追加する。

        Args:
            df (pd.DataFrame): リターン計算済みの特徴量DataFrame。
            bond_cols_for_suffix (List[str]): 該当suffixの債券列名リスト。
            suffix (str): 例: "1d_return".

        Returns:
            pd.DataFrame: 差分列を追加したDataFrame。
        """
        from collections import defaultdict
        bond_groups = defaultdict(list)

        # 債券prefix (例: "JPbond") ごとにグループ化
        for c in bond_cols_for_suffix:
            prefix, year_num = self._split_bond_prefix_and_year(c)
            if prefix and year_num is not None:
                bond_groups[prefix].append(c)

        # グループごとに全ペア差分列を作成
        for prefix, cols in bond_groups.items():
            # 年数順にソート
            sorted_cols = sorted(cols, key=lambda x: self._extract_year_num(x))
            # すべてのペア (i < j) について列を生成
            for i in range(len(sorted_cols)):
                for j in range(i+1, len(sorted_cols)):
                    bond_i = sorted_cols[i]
                    bond_j = sorted_cols[j]
                    year_i = self._extract_year_num(bond_i)
                    year_j = self._extract_year_num(bond_j)

                    # 列名例: "JPbond2_minus_5_1d_return"
                    diff_col = f"{prefix}{year_i}_minus_{year_j}_{suffix}"
                    df[diff_col] = df[bond_j] - df[bond_i]

        return df

    @staticmethod
    def _split_bond_prefix_and_year(bond_col: str):
        """
        債券列名(例: "JPbond10_1d_return")から
        prefix部分 (例: "JPbond") と 年数 (例: 10) を抽出して返す。

        Args:
            bond_col (str): 債券列名。例: "JPbond10_1d_return"

        Returns:
            (str, int or None):
                第1戻り値: prefix (例: "JPbond"), 第2戻り値: 数値の年数 (例: 10)。
                抽出できなかった場合は (None, None) を返す。
        """
        splitted = bond_col.split('_')[0]  # => "JPbond10"
        match = re.match(r'(.+bond)(\d+)$', splitted)
        if not match:
            return None, None
        prefix = match.group(1)  # "JPbond"
        year_str = match.group(2)  # "10"
        return prefix, int(year_str)

    @staticmethod
    def _extract_year_num(bond_col: str) -> int:
        """
        債券列名(例: "JPbond10_1d_return")のうち、数字部分(年数)を抜き出して返す。

        Args:
            bond_col (str): 列名。例: "JPbond10_1d_return"

        Returns:
            int: 年数 (例: 10)。正規表現で抽出できない場合は0を返す。
        """
        splitted = bond_col.split('_')[0]  # => "JPbond10"
        m = re.search(r'\d+', splitted)
        return int(m.group()) if m else 0



class PriceFeatureCalculator:
    """
    株価の時系列データからリターンやボラティリティ等を計算するクラス。

    責務:
      - 個別銘柄の株価データ (例: 'Close' 列) に対して、複数日リターンやボラティリティを計算する。
      - セクター情報や業種別の集計はここでは行わない。

    注意点:
      - リターン計算には pct_change() を用いる。
      - ボラティリティ計算には rolling(std) を用いる。
    """

    def __init__(self, yaml_path: str = Paths.STOCK_PRICE_COLUMNS_YAML, key_name: str = 'original_columns',
                 date_col: str = 'Date', code_col: str = 'Code', price_col: str = 'Close'):
        """
        コンストラクタ。

        Args:
            yaml_path (str): 列の設定を記録したYAMLファイル。
            date_col (str): 日付列の名称。デフォルトは "Date"。
            code_col (str): 日付列の名称。デフォルトは "Code"。
            price_col (str): 価格列の名称。デフォルトは "Close"。
        """
        _price_yaml = yaml_utils.including_columns_loader(yaml_path, key_name)
        self.date_col = yaml_utils.column_name_getter(_price_yaml, {'name': date_col}, 'fixed_name')
        self.code_col = yaml_utils.column_name_getter(_price_yaml, {'name': code_col}, 'fixed_name')
        self.price_col = yaml_utils.column_name_getter(_price_yaml, {'name': price_col}, 'fixed_name')

    def calculate_price_features(
        self,
        price_df: pd.DataFrame,
        return_duration: Optional[List[int]] = None,
        vola_duration: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        株価時系列に対してリターンやボラティリティを一括で計算する。

        Args:
            price_df (pd.DataFrame):
                個別銘柄の時系列DataFrame。
                インデックス・カラム構成は自由だが、少なくとも self.date_col と self.price_col が含まれる。
            return_duration (List[int] or None):
                リターンを算出する日数のリスト。None の場合はリターンを計算しない。デフォルトは[1, 5, 21]。
            vola_duration (List[int] or None):
                ボラティリティを算出する日数のリスト。None の場合はボラティリティを計算しない。デフォルトは[5, 21]。

        Returns:
            pd.DataFrame:
                計算した特徴量を列として追加したDataFrame。
                インデックス・行数は入力に準じる。
        """
        # デフォルトリストの対応（可変オブジェクトを避けるため）
        if return_duration is None:
            return_duration = [1, 5, 21]
        if vola_duration is None:
            vola_duration = [5, 21]

        # 必要であれば日付でソート
        price_df = price_df.reset_index(drop=False)
        price_df = price_df.sort_values(by=self.date_col).copy()

        # 今回の出力DF: Date, Code のみ初期列として持たせる
        df = price_df[[self.date_col, self.code_col]].copy()

        # リターン計算をヘルパー関数へ切り出し
        df = self._calculate_returns(
            df=df,
            price_df=price_df,
            return_duration=return_duration
        )

        # ボラティリティ計算をヘルパー関数へ切り出し
        df = self._calculate_volatility(
            df=df,
            price_df=price_df,
            base_return_duration=return_duration,
            vola_duration=vola_duration
        )

        return df

    # --------------------------------------------------------------------------
    #  以下、ヘルパーメソッド
    # --------------------------------------------------------------------------

    def _calculate_returns(
        self,
        df: pd.DataFrame,
        price_df: pd.DataFrame,
        return_duration: List[int]
    ) -> pd.DataFrame:
        """
        リターン(複数日数)を算出して df に列追加するヘルパーメソッド。

        Args:
            df (pd.DataFrame): 処理対象の出力用DataFrame (Date, Code など)
            price_df (pd.DataFrame): 入力の株価DataFrame
            return_duration (List[int]): リターンを算出する日数リスト

        Returns:
            pd.DataFrame: リターン列が追加された df
        """
        return_duration_sorted = sorted(return_duration)
        for d in return_duration_sorted:
            df[f"{d}d_return"] = (
                price_df.groupby(self.code_col)[self.price_col]
                .pct_change(d, fill_method=None)
                .values
            )

        return df

    def _calculate_volatility(
        self,
        df: pd.DataFrame,
        price_df: pd.DataFrame,
        base_return_duration: List[int],
        vola_duration: List[int]
    ) -> pd.DataFrame:
        """
        ボラティリティ(複数日数)を算出して df に列追加するヘルパーメソッド。

        注釈:
          - ボラティリティ計算には1日リターンが必要。
            ユーザが return_duration=[5,21] などと指定しても、
            内部的には1d_returnを一時的に計算して利用する。

        Args:
            df (pd.DataFrame): 処理対象の出力用DataFrame (Date, Code + リターン列など)
            price_df (pd.DataFrame): 入力の株価DataFrame
            base_return_duration (List[int]):
                ユーザが指定したリターン計算日数。1が含まれているかを確認。
            vola_duration (List[int]): ボラティリティを算出する日数リスト

        Returns:
            pd.DataFrame: ボラティリティ列が追加された df
        """
        # ボラティリティ計算には最低1日リターンが必要
        if 1 in base_return_duration:
            base_return_col = "1d_return"
        else:
            # 1日リターンだけ仮で計算
            df["_temp_1d_return_for_vola"] = (
                price_df.groupby(self.code_col)[self.price_col]
                .pct_change(1, fill_method=None)
                .values
            )
            base_return_col = "_temp_1d_return_for_vola"

        for v in vola_duration:
            df[f"{v}d_vola"] = (
                df.groupby(self.code_col)[base_return_col]
                  .rolling(v)

                  .std()
                  .values
            )

        # 仮列があれば削除
        if "_temp_1d_return_for_vola" in df.columns:
            df.drop(columns=["_temp_1d_return_for_vola"], inplace=True)

        return df


class FinancialFeatureCalculator:
    """
    財務情報（EPS、時価総額など）を計算するクラス。

    責務:
      - 銘柄ごとに財務データを用いて特徴量を算出（例: 平均EPS、時価総額）。
      - セクター別の集約やランキングは行わない。

    注意点:
      - 計算のために必要なカラム名やロジックは予め決めておく。
      - 時系列欠損などがある場合は適宜 ffill/bfill を行うなど運用方針に合わせる。
    """

    def __init__(self, 
                 fin_yaml_path: str = Paths.STOCK_FIN_COLUMNS_YAML,
                 sector_yaml_path: str = Paths.SECTOR_INDEX_COLUMNS_YAML):
        """
        コンストラクタ

        Args:
            yaml_path (str): カラム情報を格納したyamlファイルのパス。
        """
        _fin_yaml = yaml_utils.including_columns_loader(fin_yaml_path, 'original_columns') + \
                    yaml_utils.including_columns_loader(fin_yaml_path, 'calculated_columns')
        _sector_yaml = yaml_utils.including_columns_loader(sector_yaml_path, 'calculated_columns')
        self.date_col = yaml_utils.column_name_getter(_fin_yaml, {'name': 'DisclosedDate'}, 'fixed_name')
        self.code_col = yaml_utils.column_name_getter(_fin_yaml, {'name': 'LocalCode'}, 'fixed_name')
        self.marketcap_col = yaml_utils.column_name_getter(_sector_yaml, {'name': 'MARKET_CAP_CLOSE'}, 'fixed_name')
        self.eps_col = yaml_utils.column_name_getter(_fin_yaml, {'name': 'FORECAST_EPS'}, 'fixed_name')

    def calculate_financial_features(self, fin_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        財務情報からEPSや時価総額を計算し、銘柄×日次の形式に整形して返す。

        Args:
            fin_df (pd.DataFrame):
                財務データのDataFrame。
                例: ["Code", "Date", "ForecastEarningsPerShare", "NextYearForecastEarningsPerShare", ...]
            price_df (pd.DataFrame):
                株価データ。
                例: ["Code", "Date", "Close", "Volume", ...]
                時系列合わせや欠損埋め用に使用する。

        Returns:
            pd.DataFrame:
                銘柄×日次で EPS や 時価総額 を持つDataFrame。
                Indexやカラムは自由だが、"Date" と "Code" を含む形に整形するのが望ましい。
        """
        # 1) EPS列を合算し、ForecastEPSとして一列にまとめる
        eps_df = self._combine_forecast_eps(fin_df)
        # 2) ForecastEPSを銘柄・日付順に並べて ffill
        eps_df = self._ffill_forecast_eps(eps_df)
        # 3) price_dfから時価総額を計算 (旧コードのSectorIndexCalculatorを利用)
        market_cap_df = self._calculate_marketcap(price_df, fin_df)
        # 4) market_cap_df と eps_df をマージし、欠損を再補完
        merged = self._merge_and_fill(market_cap_df, eps_df)

        return merged

    # --------------------------------------------------------------------------
    #  以下、ヘルパーメソッド
    # --------------------------------------------------------------------------

    def _combine_forecast_eps(self, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        fin_df内のEPS列を返す。

        Args:
            fin_df (pd.DataFrame): 財務データのDataFrame。

        Returns:
            pd.DataFrame: 合算後のDataFrame（self.code_col, self.date_col, self.eps_col 列を含む）。
        """
        # 利用する列だけ残す (コード, 日付, ForecastEPS)
        return fin_df[[self.code_col, self.date_col, self.eps_col]]

    def _ffill_forecast_eps(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """
        ForecastEPS列を銘柄コード・日付順でソートし、最新値で前方埋め(ffill)する。

        Args:
            eps_df (pd.DataFrame): "ForecastEPS" 列を含むDataFrame。

        Returns:
            pd.DataFrame: ffill処理後の eps_df。
        """
        eps_df = eps_df.sort_values(by=[self.code_col, self.date_col])
        eps_df[self.eps_col] = eps_df.groupby(self.code_col)[self.eps_col].ffill()

        return eps_df

    def _calculate_marketcap(self, price_df: pd.DataFrame, fin_df: pd.DataFrame) -> pd.DataFrame:
        """
        旧コードのSectorIndexCalculator.calc_marketcap を用いて、
        price_dfからMarketCapClose列を計算したDataFrameを返す。

        Args:
            price_df (pd.DataFrame): 株価データ
            fin_df (pd.DataFrame): 財務データ (発行済株式数などが含まれている前提)

        Returns:
            pd.DataFrame:
                date_col, code_col, MarketCapCloseを含むDataFrame。
        """
        price_df_with_cap = SectorIndexCalculator.calc_marketcap(price_df, fin_df)

        # marketcap_col が計算された場合、列名を固定 (MarketCapClose) に変更
        if self.marketcap_col in price_df_with_cap.columns:
            market_cap_df = price_df_with_cap[[self.date_col, self.code_col, self.marketcap_col]].copy()
            market_cap_df.rename(columns={self.marketcap_col: "MarketCapClose"}, inplace=True)
        else:
            # 想定外の場合、空のDataFrameを返すかエラーを投げるなど運用次第
            market_cap_df = pd.DataFrame(columns=[self.date_col, self.code_col, "MarketCapClose"])

        return market_cap_df.sort_values(by=[self.date_col, self.code_col])

    def _merge_and_fill(self, market_cap_df: pd.DataFrame, eps_df: pd.DataFrame) -> pd.DataFrame:
        """
        時価総額データ(market_cap_df)とEPSデータ(eps_df)をマージし、欠損をffill/bfillする。

        Args:
            market_cap_df (pd.DataFrame): "Date", "Code", "MarketCapClose" 列を含むDataFrame
            eps_df (pd.DataFrame): "Date", "Code", "ForecastEPS" 列を含むDataFrame

        Returns:
            pd.DataFrame:
                マージ後、欠損値を埋めたDataFrame。
                "Date", "Code", "MarketCapClose", "ForecastEPS" を含む。
        """
        merged = pd.merge(
            market_cap_df, eps_df,
            on=[self.date_col, self.code_col],
            how="left"
        ).sort_values(by=[self.date_col, self.code_col])

        # 欠損補完
        merged["ForecastEPS"] = merged.groupby(self.code_col)["ForecastEPS"].ffill()
        merged["ForecastEPS"] = merged.groupby(self.code_col)["ForecastEPS"].bfill()
        if "MarketCapClose" in merged.columns:
            merged["MarketCapClose"] = merged.groupby(self.code_col)["MarketCapClose"].ffill()

        return merged


class FeaturesMerger:
    """
    複数の特徴量DataFrameをマージ・結合するクラス。

    責務:
      - DateやCodeなど、指定されたキー列で単純な結合を行う。
      - マージ後の欠損値補完と行方向でのドロップを行う。
      - 最終的に欠損値の残らないDataFrameを返す。
    """

    def __init__(
        self,
        date_col: str = "Date",
        code_col: str = "Code",
        how: str = "outer"
    ):
        """
        コンストラクタ

        Args:
            date_col (str): 日付列の名称。デフォルトは "Date"。
            code_col (str): 銘柄コード列の名称。デフォルトは "Code"。
            how (str): データ結合の方法。デフォルトは "outer"。
                       例: "left", "right", "inner", "outer".
        """
        self.date_col = date_col
        self.code_col = code_col
        self.how = how

    def merge_on_date(self, df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Date列のみをキーとして、複数のDataFrameを順次マージする。

        結合後:
          1. Dateでソート
          2. Date列を左端に配置 (Codeがあっても無視)
          3. ffill & dropna

        Args:
            df_list (List[pd.DataFrame]): 
                結合対象となるDataFrameのリスト。
                すべてに self.date_col が存在する必要がある。

        Returns:
            pd.DataFrame:
                Merge後に欠損補完とdropnaを実施した最終DataFrame。
        """
        if not df_list:
            return pd.DataFrame()

        merged_df = df_list[0].copy()
        for i in range(1, len(df_list)):
            merged_df = pd.merge(
                merged_df,
                df_list[i],
                on=self.date_col,
                how=self.how
            )

        # 1. ソート
        merged_df = merged_df.sort_values(by=self.date_col).reset_index(drop=True)
        # 2. ffill → dropna
        merged_df = self._fill_and_dropna(merged_df)
        # 3. Date列 (Codeがあればそれも) を左端に
        merged_df = self._move_keycols_to_left(merged_df)

        return merged_df

    def merge_on_date_code(self, df_list: List[pd.DataFrame]) -> pd.DataFrame:
        """
        複数のDataFrameを順次マージし、最終的にDate, Codeをキーとした
        形での結合を試みる。ただし、片方がCode列を持たない場合はDateのみで結合。

        結合後:
          1. Date(+Code) でソート
          2. Date, Codeを左端に移動 (Code列があれば)
          3. ffill → dropna

        Args:
            df_list (List[pd.DataFrame]):
                結合対象のDataFrameリスト。
                いずれも self.date_col を持つ必要がある。
                Code列は存在してもしなくても良い。

        Returns:
            pd.DataFrame:
                Merge後に欠損補完とdropnaを実施した最終DataFrame。
        """
        if not df_list:
            return pd.DataFrame()

        merged_df = df_list[0].copy()

        for i in range(1, len(df_list)):
            next_df = df_list[i].copy()

            left_has_code = self.code_col in merged_df.columns
            right_has_code = self.code_col in next_df.columns

            if left_has_code and right_has_code:
                # 両方にCode列がある → Date, Code で結合
                merged_df = pd.merge(
                    merged_df,
                    next_df,
                    on=[self.date_col, self.code_col],
                    how=self.how
                )
            else:
                # 片方または両方にCode列がない → Dateのみで結合
                merged_df = pd.merge(
                    merged_df,
                    next_df,
                    on=self.date_col,
                    how=self.how
                )

        # ソート
        if self.code_col in merged_df.columns:
            merged_df = merged_df.sort_values(by=[self.date_col, self.code_col]).reset_index(drop=True)
        else:
            merged_df = merged_df.sort_values(by=self.date_col).reset_index(drop=True)

        # ffill + dropna
        merged_df = self._fill_and_dropna(merged_df)
        # Date, Code 列を左端へ
        merged_df = self._move_keycols_to_left(merged_df)

        return merged_df

    def merge_on_any_keys(self, df_list: List[pd.DataFrame], keys: List[str]) -> pd.DataFrame:
        """
        任意のキー（複数列）を指定して順次マージする。

        結合後:
          1. keysでソート
          2. keys(特にDate, Code含む場合) を左端に移動
          3. ffill → dropna

        Args:
            df_list (List[pd.DataFrame]):
                結合対象のDataFrameリスト。
            keys (List[str]): 
                結合キーとなる列名のリスト。

        Returns:
            pd.DataFrame:
                keysでマージ後に欠損補完とdropnaを実施した最終DataFrame。
        """
        if not df_list or not keys:
            return pd.DataFrame()

        merged_df = df_list[0].copy()
        for i in range(1, len(df_list)):
            merged_df = pd.merge(
                merged_df,
                df_list[i],
                on=keys,
                how=self.how
            )

        # ソート
        merged_df = merged_df.sort_values(by=keys).reset_index(drop=True)
        # ffill + dropna
        merged_df = self._fill_and_dropna(merged_df)
        # keys(特にDate, Code)を左端へ
        merged_df = self._move_keycols_to_left(merged_df, keys=keys)

        return merged_df

    # ----------------------------------------------------------------------
    #  ヘルパー関数: キー列(DATE, CODE)を左端へ並べ替える
    # ----------------------------------------------------------------------
    def _move_keycols_to_left(self, df: pd.DataFrame, keys: List[str] = None) -> pd.DataFrame:
        if keys is None:
            # Date, Code をキーとして想定
            keys = []
            if self.date_col in df.columns:
                keys.append(self.date_col)
            if self.code_col in df.columns:
                keys.append(self.code_col)

        unique_cols = [col for col in keys if col in df.columns]
        other_cols = [col for col in df.columns if col not in unique_cols]
        new_col_order = unique_cols + other_cols

        return df[new_col_order]

    # ----------------------------------------------------------------------
    # ヘルパー関数: ffill + dropna (Code列を保持する)
    # ----------------------------------------------------------------------
    def _fill_and_dropna(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) ffillで欠損値を埋める (コードがあれば銘柄単位)
        2) dropna(how="any") で行方向の欠損を削除

        Returns:
            pd.DataFrame: 'Code' 列を残したまま欠損なしの状態へ
        """
        if self.code_col in df.columns:
            # グループ化してffillし、Code列を保持
            df = (
                df.groupby(self.code_col, group_keys=False)
                  .apply(lambda g: g.ffill(), include_groups=True)
            )
        else:
            # Code列がなければ全体でffill
            df = df.ffill()

        # dropna
        df = df.dropna(how="any")
        return df



# セクターを設定しない場合の使用方法
if __name__ == '__main__':
    from utils.paths import Paths
    from datetime import datetime
    
    setting_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
    ifc = IndexFeatureCalculator()
    for _, row in setting_df.iterrows():
        fmd = FeatureMetadata(row['Name'], row['Group'], f"{Paths.SCRAPED_DATA_FOLDER}/{row['Group']}/{row['Path']}", row['URL'], row['is_adopted'])
        ifc.calculate_return(feature_metadata=fmd, days=1)
    index_df = ifc.finalize()

    from facades.stock_acquisition_facade import StockAcquisitionFacade
    acq = StockAcquisitionFacade(filter = "(Listing==1)&((ScaleCategory=='TOPIX Core30')|(ScaleCategory=='TOPIX Large70')|(ScaleCategory=='TOPIX Mid400'))")
    stock_dfs = acq.get_stock_data_dict()
    pfc = PriceFeatureCalculator()
    return_df = pfc.calculate_price_features(stock_dfs['price'])
    
    ffc = FinancialFeatureCalculator()
    financial_df = ffc.calculate_financial_features(stock_dfs['fin'], stock_dfs['price'])
    
    fm = FeaturesMerger()
    merged_df = fm.merge_on_date_code([index_df, return_df, financial_df])
    print(merged_df)

# セクターを設定する場合の使用方法
if __name__ == '__main__':
    from utils.paths import Paths
    from datetime import datetime
    
    setting_df = pd.read_csv(Paths.FEATURES_TO_SCRAPE_CSV)
    ifc = IndexFeatureCalculator()
    for _, row in setting_df.iterrows():
        fmd = FeatureMetadata(row['Name'], row['Group'], f"{Paths.SCRAPED_DATA_FOLDER}/{row['Group']}/{row['Path']}", row['URL'], row['is_adopted'])
        ifc.calculate_return(feature_metadata=fmd, days=1)
    index_df = ifc.finalize()

    sector_index_df = pd.read_parquet(f'{Paths.SECTOR_PRICE_FOLDER}/New48sectors_price.parquet')
    pfc = PriceFeatureCalculator(yaml_path=Paths.SECTOR_INDEX_COLUMNS_YAML, key_name='calculated_columns', date_col='DATE', code_col='SECTOR', price_col='CLOSE')
    return_df = pfc.calculate_price_features(sector_index_df)
    
    financial_df = pd.read_parquet(f'{Paths.SECTOR_FIN_FOLDER}/New48sectors_fin.parquet').reset_index(drop=False)
       
    fm = FeaturesMerger(code_col='Sector')
    merged_df = fm.merge_on_date_code([index_df, return_df, financial_df])
    print(merged_df)