import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import open3d as o3d
from networks import student_net, teacher_net
from PIL import Image
from torchvision import transforms
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)
    
def visualize_anomalies(data_generator):
    for val, (point_cloud, anomaly_scores) in enumerate(data_generator):
        reshaped_anomaly_scores = anomaly_scores.reshape(-1)  # (batch_size * num_points)
        normalized_scores = (reshaped_anomaly_scores - reshaped_anomaly_scores.min()) / (reshaped_anomaly_scores.max() - reshaped_anomaly_scores.min())
        colors = plt.cm.jet(normalized_scores)[:, :3] 
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([pcd])
        break

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def normalize_scores(scores, exponent = 1):
    return (scores - scores.min())**exponent / (scores.max() - scores.min())**exponent

def compute_anomaly_scores(student, teacher, dataloader, device):
    teacher.eval()
    student.eval()
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            scores = torch.norm(teacher(data) - student(data), dim=2)
            point_cloud = data.pos.cpu().numpy()
            anomaly_scores = scores.cpu().numpy()
            reshaped_point_cloud = point_cloud.reshape(262144, 3)  # (batch_size * num_points, 3)
            reshaped_anomaly_scores = normalize_scores(anomaly_scores.reshape(-1)) 
            reshaped_array = np.zeros((262144, 3)) # Create arrays for x, y, and z coordinates

            # Iterate over each index and assign its position as the value
            for i in range(262144):
                reshaped_array[i] = [i% 512, i // 512, reshaped_point_cloud[i][0]-reshaped_point_cloud[i][1]]
            yield reshaped_array, reshaped_anomaly_scores
    return results

if __name__ == "__main__":
    filename = "testimg.png"
    if len(sys.argv) != 2:
        print("Running on default file: testimg.png")
    else:
        filename = sys.argv[1]
        print("Running test on filename: ", filename)
    print("Started")
    
    feature_dim = 64
    batch_size = 1
    fixed_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    if str(device) == "cpu":
        print("Not using GPU, exiting.")
        exit()

    torch.multiprocessing.set_start_method('spawn', True)
    
    print("Initializing networks...")
    teacher_path = 'teacher_model.pth'
    student_path = 'student_model.pth'
    
    initialized_teacher = teacher_net.TeacherNet(feature_dim).to(device)
    initialized_student = student_net.StudentNet(feature_dim).to(device)

    resizer = transforms.Compose([transforms.Resize((fixed_size,fixed_size))])

    img = Image.open(filename)
    img = resizer(img)
    points = np.array(img)
    print("Image shape: "+str(points.shape))
    pos = torch.tensor(points, dtype=torch.float)
    print("POS shape: "+str(pos.size()))
    if (pos.size()==torch.Size([512,512])):
        pos = pos.unsqueeze(-1).repeat(1, 1, 3)
    data = Data(pos=pos)
    test_loader = DataLoader([data], batch_size=batch_size, shuffle=False, num_workers=0)

    initialized_teacher = load_model(initialized_teacher, teacher_path, device)
    initialized_student = load_model(initialized_student, student_path, device)
    
    print("Evaluating student and teacher networks...")
    results = compute_anomaly_scores(initialized_student, initialized_teacher, test_loader, device)
    visualize_anomalies(results)
