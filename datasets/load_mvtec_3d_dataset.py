import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import json
from PIL import Image


class MVTec3D(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None):
        self.split = split
        super(MVTec3D, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'{self.split}_data.pt']

    def process(self):
        data_list = []

        # Directory for the specific split
        split_dir = os.path.join(self.root, 'mvtec_3d_anomaly_detection')
        print(f"Processing split: {self.split}")
        
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset directory not found: {split_dir}")
        
        # Iterate over each object type directory (bagel, carrot, etc.)
        for obj_type in os.listdir(split_dir):
            obj_dir = os.path.join(split_dir, obj_type, self.split)
            if not os.path.isdir(obj_dir):
                continue

            print(f"Processing object type: {obj_type}")

            # Traverse through subfolders (Combined, Contamination, Crack, etc.)
            for subfolder in os.listdir(obj_dir):
                subfolder_dir = os.path.join(obj_dir, subfolder)
                if not os.path.isdir(subfolder_dir):
                    continue

                # Process each file in the subfolders (gt, rgb, xyz)
                for folder_name in os.listdir(subfolder_dir):
                    folderstr = subfolder_dir+"\\"+folder_name
                    print(f"Found folder: {folderstr}")  # Debug: Print out the file names
                    for file_name in os.listdir(folderstr):
                        file_path = os.path.join(subfolder_dir, folder_name+"\\"+file_name)
                        print(f"Found file: {file_name}")  # Debug: Print out the file names
                        if file_name.endswith('.tiff'):  # Update the file extension to '.tiff'
                            print(f"Processing TIFF file: {file_path}")
                            
                            # Load the data from .tiff file using PIL
                            img = Image.open(file_path)
                            points = np.array(img)  # Convert image to numpy array
                            pos = torch.tensor(points, dtype=torch.float)
                            
                            # Create Data object
                            data = Data(pos=pos)
                            data_list.append(data)
                        elif file_name.endswith('.png'):
                            print(f"Processing PNG file: {file_path}")
                            img = Image.open(file_path)
                            points = np.array(img)  # Convert image to numpy array
                            pos = torch.tensor(points, dtype=torch.float)

        if len(data_list) == 0:
            raise RuntimeError("No data found in the specified directories.")
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


