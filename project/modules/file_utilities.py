import os
import paths

def get_newest_two_files(directory: str) -> tuple[str, str]:
    # ディレクトリ内のすべてのファイルとサブディレクトリを取得
    files = os.listdir(directory)
    # ファイルがない場合はNoneを返す
    if not files:
        return None, None
    # ファイルのフルパスを取得
    full_paths = [os.path.join(directory, f) for f in files]
    # 保存日時がもっとも新しいファイルを見つける
    newest_file = max(full_paths, key=os.path.getmtime)
    if len(files) > 1:
        second_newest_file = sorted(full_paths, key=os.path.getmtime, reverse=True)[1]
    else:
        second_newest_file = None

    return newest_file, second_newest_file