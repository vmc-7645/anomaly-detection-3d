import os
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import open3d as o3d
from datasets import MVTec3D
from networks import StudentNet, TeacherNet

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
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            features = teacher(data)
            reconstructions = decoder(features)
            loss = loss_fn(reconstructions, data.pos)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

def train_student(student, teacher, dataloader, device, epochs=10):
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            data = data.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_features = teacher(data)
            student_features = student(data)
            loss = loss_fn(student_features, teacher_features)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}')

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
    # Normalize anomaly scores for visualization
    normalized_scores = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min())
    
    # Create color map based on normalized scores
    colors = plt.cm.jet(normalized_scores)[:, :3]
    
    # Visualize using Open3D
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    print("Started")
    
    # Parameters
    feature_dim = 64
    batch_size = 16
    epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # Initialize networks
    initialized_teacher = TeacherNet(feature_dim).to(device)
    initialized_student = StudentNet(feature_dim).to(device)
    decoder = DecoderNet(feature_dim).to(device)
    
    # Load dataset and prepare DataLoader
    train_dataset = MVTec3D(root='./datasets', split='train')
    test_dataset = MVTec3D(root='./datasets', split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Pretrain the teacher network
    pretrain_teacher(initialized_teacher, decoder, train_loader, device, epochs)
    
    # Train the student network
    train_student(initialized_student, initialized_teacher, train_loader, device, epochs)
    
    # Compute anomaly scores on the test set
    anomaly_scores_list = compute_anomaly_scores(initialized_student, initialized_teacher, test_loader, device)
    
    # Assuming you want to visualize anomalies for a specific test sample
    sample_idx = 0  # Change this index to visualize different samples
    test_sample = test_dataset[sample_idx]
    anomaly_scores = anomaly_scores_list[sample_idx]
    
    visualize_anomalies(test_sample.pos.numpy(), anomaly_scores)
