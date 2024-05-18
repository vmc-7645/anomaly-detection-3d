import os
import open3d as o3d
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

class MVTec3D(InMemoryDataset):
    def __init__(self, root, train=True, transform=None, pre_transform=None):
        self.train = train
        super(MVTec3D, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        dataset_path = os.path.join(self.root, 'train' if self.train else 'test')
        
        for class_dir in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_dir)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    if file_name.endswith('.ply'):
                        file_path = os.path.join(class_path, file_name)
                        pcd = o3d.io.read_point_cloud(file_path)
                        points = torch.tensor(pcd.points, dtype=torch.float)
                        data = Data(pos=points)
                        data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def load_mvtec_3d_dataset(train=True):
    dataset = MVTec3D(root='dataset', train=train)
    return dataset

# Example usage
if __name__ == "__main__":
    train_dataset = load_mvtec_3d_dataset(train=True)
    test_dataset = load_mvtec_3d_dataset(train=False)
