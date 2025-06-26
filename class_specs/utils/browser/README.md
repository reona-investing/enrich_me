# utils/browser のクラス仕様書

## browser_manager.py

### class BrowserManager
ブラウザセッション全体を管理し、NamedTab を用いて複数タブを名前で管理するクラス
- __init__: 
- get_tab: 指定した名前のタブを取得する。

Args:
    name (str): 取得したいタブの識別名。

Returns:
    NamedTab | None: 取得した NamedTab インスタンス。存在しない場合は None。
- rename_tab: 指定したタブの名前を変更する。

Args:
    name (str): 変更前のタブ名。
    new_name (str): 変更後のタブ名。

## browser_utils.py

### class BrowserUtils
ブラウザ全体の基本操作（起動、再利用、リセットなど）を提供するクラス

## named_tab.py

### class NamedTab
タブの名前とその管理オブジェクト（TabManager）を保持するデータクラス

## tab_manager.py

### class TabManager
タブのライフサイクル管理を行い、内部で TabUtils を保持するクラス
- __init__: 
- tab: nodriverの Tab インスタンスを返す
- utils: タブ上の操作を提供する TabUtils インスタンスを返す

## tab_utils.py

### class TabUtils
nodriver の Tab オブジェクトに対する操作（URLオープン、リロード、クリック等）をラップするユーティリティクラス
Node document エラー対応版
- __init__: Args:
    tab (nodriver.Tab): nodriverのタブオブジェクト

