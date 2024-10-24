import subprocess
# 依存関係を修正し、必要なパッケージをインストール
commands = [
    "apt --fix-broken install -y",
    "apt-get update -qq",
    "apt-get install -qq -y xvfb"
]

# 各コマンドを実行
for cmd in commands:
    subprocess.run(cmd, shell=True)

# 仮想ディスプレイのセットアップ
from pyvirtualdisplay import Display
# 仮想ディスプレイを開始
display = Display(visible=0, size=(1024, 768))
display.start()