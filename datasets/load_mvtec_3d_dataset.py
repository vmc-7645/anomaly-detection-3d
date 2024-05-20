import os
import torch
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
import numpy as np
import json
from PIL import Image
from torchvision import transforms

from skimage import io
# import matplotlib.pyplot as plt


class MVTec3D(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, fixed_size=1024):
        self.split = split
        self.fixed_size = fixed_size
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
        resizer = transforms.Compose([transforms.Resize((self.fixed_size,self.fixed_size))])

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
                    # print(f"Found folder: {folderstr}")  # Debug: Print out the file names
                    # if "foam" in folderstr:
                    #     print("Early exit")
                    #     break
                    for file_name in os.listdir(folderstr):
                        file_path = os.path.join(subfolder_dir, folder_name+"\\"+file_name)
                        if file_name.endswith('.tiff'):  # Update the file extension to '.tiff'
                            img = io.imread(file_path)
                            img = Image.fromarray(img[0])
                            img = resizer(img)
                            points = np.array(img)  # Convert image to numpy array
                            pos = torch.tensor(points, dtype=torch.float).unsqueeze(-1).repeat(1, 1, 3)
                        elif file_name.endswith('.png'):
                            img = Image.open(file_path)
                            img = resizer(img)
                            points = np.array(img)  # Convert image to numpy array
                            pos = torch.tensor(points, dtype=torch.float)
                        if (pos.size()==torch.Size([512,512])):
                            pos = pos.unsqueeze(-1).repeat(1, 1, 3)
                        # print(str(file_path)+" : "+str(pos.size()))
                        data = Data(pos=pos)
                        data_list.append(data)
                        

        if len(data_list) == 0:
            raise RuntimeError("No data found in the specified directories.")
        
        print("Saving data...")
        try:
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[0])
            print(f"Data successfully saved to {self.processed_paths[0]}")
        except Exception as e:
            print(f"An error occurred: {e}")
            # Detailed logging
            import traceback
            traceback.print_exc()


