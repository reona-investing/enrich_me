import pandas as pd
from datetime import datetime
import paths
from jquants_api_operations.utils import FileHandler
from jquants_api_utils import cli
from FlagManager import flag_manager, Flags

def update_price(
    basic_path: str = paths.RAW_STOCK_PRICE_PARQUET
) -> None:
    """
    株価情報を更新し、指定されたパスにParquet形式で保存します。
    :param basic_path: パスのテンプレート（例: "path_to_data/0000.parquet"）
    """
    current_year = datetime.today().year
    prev_year = current_year - 1

    current_year_path = _generate_file_path(current_year, basic_path)
    prev_year_path = _generate_file_path(prev_year, basic_path)

    # 今年のデータが存在しない場合、前年データを更新
    if not FileHandler.file_exists(current_year_path):
        _update_yearly_stock_price(prev_year, prev_year_path)

    # 今年のデータを取得・更新
    raw_stock_price = _update_yearly_stock_price(current_year, current_year_path)

    print(raw_stock_price.tail(2))


def _generate_file_path(year: int, basic_path: str) -> str:
    """指定された年に基づいてファイルパスを生成します。"""
    return basic_path.replace('0000', str(year))


def _update_yearly_stock_price(year: int, yearly_path: str) -> pd.DataFrame:
    """
    特定の年の株価情報を取得し、更新します。
    :param year: 対象の年
    :param yearly_path: 年ごとのファイルパス
    """
    # ファイルが存在すれば読み込み
    existing_data = FileHandler.read_parquet(yearly_path) if FileHandler.file_exists(yearly_path) else pd.DataFrame()

    # データ更新
    updated_data = _update_yearly_price(year, existing_data)

    # 重複排除とクリーンアップ
    cleaned_data = updated_data.drop_duplicates().dropna()

    # 更新されたデータを保存
    FileHandler.write_parquet(cleaned_data, yearly_path)
    return cleaned_data


def _update_yearly_price(year: int, existing_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    年次データを取得し、既存データを更新します。
    :param year: 対象の年
    :param existing_data: 既存の価格情報データ
    """
    if existing_data.empty:
        fetched_data = _fetch_full_year_stock_price(year)
        _set_adjustment_flag(fetched_data)
        return fetched_data

    last_exist_date = pd.to_datetime(existing_data["Date"]).iat[-1].date()
    if last_exist_date != datetime.today().date():
        new_data = _fetch_new_stock_price(last_exist_date)
        _set_adjustment_flag(new_data)
        return _update_raw_stock_price(existing_data, new_data)

    return existing_data


def _fetch_full_year_stock_price(year: int) -> pd.DataFrame:
    """指定された年の全期間の価格情報を取得します。"""
    return cli.get_price_range(
        start_dt=datetime(year, 1, 1),
        end_dt=datetime(year, 12, 31)
    )


def _fetch_new_stock_price(last_exist_date: datetime) -> pd.DataFrame:
    """最新の日付までの新しい価格情報を取得します。"""
    return cli.get_price_range(
        start_dt=last_exist_date,
        end_dt=datetime.today()
    )


def _set_adjustment_flag(fetched_stock_price: pd.DataFrame):
    """AdjustmentFactorが変更された場合のフラグを設定します。"""
    if any(fetched_stock_price['AdjustmentFactor'] != 1):
        flag_manager.set_flag(Flags.PROCESS_STOCK_PRICE, True)


def _update_raw_stock_price(existing_data: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """既存の価格情報に新しいデータを追加し、重複を削除します。"""
    combined_data = pd.concat([existing_data, new_data], axis=0)
    combined_data["Date"] = pd.to_datetime(combined_data["Date"])
    return combined_data[
        combined_data['Code'].notnull()
    ].drop_duplicates(subset=['Date', 'Code'])


if __name__ == '__main__':
    update_price()
