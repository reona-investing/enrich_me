import traceback
import csv

def get_traceback():
    '''
    エラーのトレースバックを取得
    '''
    formatted_exc = traceback.format_exc() # str を返す
    return formatted_exc.splitlines() # 各行をリストの要素として取得

def handle_exception(csv_path: str):
    # 発生中の例外に関する情報を取得する
    formatted_exc_lines = get_traceback()
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        for line in formatted_exc_lines:
            csvwriter.writerow([line])