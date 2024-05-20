# import os
from os import path
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import open3d as o3d
from datasets.load_mvtec_3d_dataset import MVTec3D
from networks import student_net, teacher_net
from numpy import concatenate

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)

class DecoderNet(nn.Module):
    def __init__(self, feature_dim):
        super(DecoderNet, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.decoder(x)

def pretrain_teacher(teacher, decoder, dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(list(teacher.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_fn = nn.MSELoss()
    teacher.train()
    decoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            features = teacher(data)
            reconstructions = decoder(features)
            loss = loss_fn(reconstructions, data.pos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Teacher: Epoch {epoch + 1}, Batch {i + 1}/{len(dataloader)}, Loss: {loss.item()}')
        print(f'Teacher: Epoch {epoch + 1}, Avg Loss: {total_loss / len(dataloader)}')


def train_student(student, teacher, dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for i, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_features = teacher(data)
            student_features = student(data)
            loss = loss_fn(student_features, teacher_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Student: Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item()}')
        print(f'Student: Epoch {epoch}, Avg Loss: {total_loss / len(dataloader)}')

def compute_anomaly_scores(student, teacher, dataloader, device):
    teacher.eval()
    student.eval()
    anomaly_scores = []
    for data in dataloader:
        data = data.to(device)
        with torch.no_grad():
            teacher_features = teacher(data)
            student_features = student(data)
            scores = torch.norm(teacher_features - student_features, dim=1)
            anomaly_scores.append(scores.cpu().numpy())
    return anomaly_scores

def visualize_anomalies(point_cloud, anomaly_scores):
    normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    colors = plt.cm.jet(normalized_scores)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

if __name__ == "__main__":
    print("Started")
    
    # Parameters
    feature_dim = 64
    batch_size = 2
    epochs = 8 # 10-11 ideal
    fixed_size = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.multiprocessing.set_start_method('spawn', True)
    print(device)
    if str(device) == "cpu":
        print("Not using GPU, exiting.")
        exit
    
    # Initialize networks
    print("Initializing networks...")
    teacher_path = 'teacher_model.pth'
    student_path = 'student_model.pth'
    decoder_path = 'decoder_model.pth'
    
    initialized_teacher = teacher_net.TeacherNet(feature_dim).to(device)
    initialized_student = student_net.StudentNet(feature_dim).to(device)
    decoder = DecoderNet(feature_dim).to(device)
    
    # Load dataset and prepare DataLoader
    print("Loading train data...")
    train_dataset = MVTec3D(root='./datasets', split='train', fixed_size=fixed_size)
    print(f'Train dataset size: {len(train_dataset)}')

    print("Loading test data...")
    test_dataset = MVTec3D(root='./datasets', split='test', fixed_size=fixed_size)
    print(f'Test dataset size: {len(test_dataset)}')

    print("Preparing dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Load models if they exist
    if path.exists(teacher_path) and path.exists(student_path) and path.exists(decoder_path):
        print("Loading saved models...")
        initialized_teacher = load_model(initialized_teacher, teacher_path, device)
        initialized_student = load_model(initialized_student, student_path, device)
        decoder = load_model(decoder, decoder_path, device)
    else:
        # Pretrain the teacher network
        print("Pretraining teacher...")
        pretrain_teacher(initialized_teacher, decoder, train_loader, device, epochs)
        save_model(initialized_teacher, teacher_path)
        save_model(decoder, decoder_path)
        
        # Train the student network
        print("Training student...")
        train_student(initialized_student, initialized_teacher, train_loader, device, epochs)
        save_model(initialized_student, student_path)
    
    # Compute anomaly scores on the test set
    print("Computing anomaly scores...")
    anomaly_scores_list = compute_anomaly_scores(initialized_student, initialized_teacher, test_loader, device)
    
    # Visualize anomalies
    print("Visualizing anomalies...")
    test_point_cloud = next(iter(test_loader)).pos.numpy()
    anomaly_scores = concatenate(anomaly_scores_list)
    visualize_anomalies(test_point_cloud, anomaly_scores)