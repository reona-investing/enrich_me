import unittest
import os
import sys

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """全てのテストを実行する"""
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(start_dir=os.path.dirname(__file__), pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result

if __name__ == '__main__':
    result = run_all_tests()
    
    # テスト結果のサマリを表示
    print("\nテスト結果サマリ:")
    print(f"実行数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失敗: {len(result.failures)}")
    print(f"エラー: {len(result.errors)}")
    
    # 終了コードを設定
    sys.exit(len(result.failures) + len(result.errors))