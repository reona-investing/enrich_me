install_script = """
import os
import subprocess
import re

#content直下にchrome関係ファイルをインストールするため
current_dir = os.getcwd()
os.chdir('/content')

#stable versionのページを開く
chrome_driver_version = subprocess.check_output(['curl', '-sS', 'https://googlechromelabs.github.io/chrome-for-testing/#stable'], text=True).strip()
#URLのパターンから、stable versionを取得
pattern = r'https://storage.googleapis.com/chrome-for-testing-public/(\d+\.\d+\.\d+\.\d+)/linux64/chrome-linux64\.zip'
matches = re.findall(pattern, chrome_driver_version) # 正規表現によるマッチング
stable_version = matches[0]

# stable versionのダウンロードURLを取得
chrome_url = f'https://storage.googleapis.com/chrome-for-testing-public/{stable_version}/linux64/chrome-linux64.zip'
driver_url = f'https://storage.googleapis.com/chrome-for-testing-public/{stable_version}/linux64/chromedriver-linux64.zip'

# 更新を実行
subprocess.run(['sudo', 'apt', '-y', 'update'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# ダウンロードのために必要なパッケージをインストール
subprocess.run(['sudo', 'apt', 'install', '-y', 'wget', 'curl', 'unzip'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# Chromeの依存パッケージをインストール
subprocess.run(['wget', 'http://archive.ubuntu.com/ubuntu/pool/main/libu/libu2f-host/libu2f-udev_1.1.4-1_all.deb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
subprocess.run(['dpkg', '-i', 'libu2f-udev_1.1.4-1_all.deb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
# Chromeのインストール
subprocess.run(['wget', 'https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
subprocess.run(['dpkg', '-i', 'google-chrome-stable_current_amd64.deb'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

os.chdir(current_dir)
"""

code_obj = compile(install_script, '<string>', 'exec')