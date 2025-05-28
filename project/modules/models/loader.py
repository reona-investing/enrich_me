from typing import List, Tuple
from models import MLDataset

def load_datasets(*dataset_paths: str, ensembled_dataset_path: str, should_learn: bool = False) -> Tuple[List[MLDataset], MLDataset]:
    """
    任意個数のデータセットパスと、アンサンブル用のデータセットパスを受け取り、
    MLDatasetオブジェクトのリストとアンサンブル用MLDatasetオブジェクトを返す関数。
    
    使用例:
        datasets, ensembled = load_datasets("path1", "path2", "path3", ensembled_dataset_path="ensembled_path")
    """
    if should_learn:
        datasets = [MLDataset(path, init_load=False) for path in dataset_paths]
        ensembled_dataset = MLDataset(ensembled_dataset_path, init_load=False)
    else:
        datasets = [MLDataset(path) for path in dataset_paths]
        ensembled_dataset = MLDataset(ensembled_dataset_path)
    
    return datasets, ensembled_dataset
