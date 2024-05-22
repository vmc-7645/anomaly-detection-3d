import os
from os import path
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import open3d as o3d
from datasets.load_mvtec_3d_dataset import MVTec3D
from networks import student_net, teacher_net, decoder_net
import numpy as np
from numpy import concatenate, ndarray

import multiprocessing as mp
mp.set_start_method('spawn', force=True)
torch.set_num_threads(1)
torch.autograd.set_detect_anomaly(True)


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
            if i % 100 == 0:print(f'Teacher: Epoch {epoch + 1}, Batch {i + 1}/{len(dataloader)}, Loss: {loss.item()}')
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
            if i % 100 == 0:print(f'Student: Epoch {epoch}, Batch {i}/{len(dataloader)}, Loss: {loss.item()}')
        print(f'Student: Epoch {epoch}, Avg Loss: {total_loss / len(dataloader)}')

# Function to compute anomaly scores and save them to a file
def compute_anomaly_scores(student, teacher, dataloader, device):
    teacher.eval()
    student.eval()
    v = "0.0, 0.0, 0.0"
    with torch.no_grad():
        with open('anomaly_scores.txt', 'w') as f:
            for i, data in enumerate(dataloader):
                data = data.to(device)
                print(f"Batch {i}:")
                print(f"Positions: {data.pos.cpu().numpy()[:5]}")  # Print first 5 positions
                teacher_features = teacher(data)
                student_features = student(data)
                scores = torch.norm(teacher_features - student_features, dim=2)

                point_cloud = data.pos.cpu().numpy()
                anomaly_scores = scores.cpu().numpy()

                # Debugging: Print the first few points and scores before reshaping
                print("Before Reshaping:")
                print(f"Point Cloud Shape: {str(point_cloud.shape)}")
                print(f"Point Cloud Data: {point_cloud[:5]}")
                print(f"Anomaly Scores: {anomaly_scores[:5]}")

                # Reshape point cloud and anomaly scores
                reshaped_point_cloud = point_cloud.reshape(-1, 3)  # (batch_size * num_points, 3)
                reshaped_anomaly_scores = anomaly_scores.reshape(-1)  # (batch_size * num_points)

                # Debugging: Print the first few points and scores after reshaping
                print("After Reshaping:")
                print(f"Reshaped Point Cloud Data: {reshaped_point_cloud[:5]}")
                print(f"Reshaped Anomaly Scores: {reshaped_anomaly_scores[:5]}")

                # Write to file
                for pc, score in zip(reshaped_point_cloud, reshaped_anomaly_scores):
                    b = f"{pc[0]}, {pc[1]}, {pc[2]}"
                    if b!=v:f.write(f"{b}, {score}\n")

                yield reshaped_point_cloud, reshaped_anomaly_scores
    
def visualize_anomalies(data_generator):
    for val, (point_cloud, anomaly_scores) in enumerate(data_generator):
        if val < 10 :
            continue
        print(f"{val}: PointCloud Shape: {point_cloud.shape}, Anomaly Scores Shape: {anomaly_scores.shape}")

        # Check the initial shapes
        if point_cloud.shape[2] != 3:
            raise ValueError("PointCloud should have shape (batch_size, num_points, 3)")

        if anomaly_scores.shape[1] != point_cloud.shape[1]:
            raise ValueError("Anomaly Scores should have the same number of points as PointCloud")

        # Reshape point cloud and anomaly scores
        batch_size, num_points, _ = point_cloud.shape
        reshaped_point_cloud = point_cloud.reshape(-1, 3)  # (batch_size * num_points, 3)
        reshaped_anomaly_scores = anomaly_scores.reshape(-1)  # (batch_size * num_points)

        print(f"Reshaped PointCloud Shape: {reshaped_point_cloud.shape}")
        print(f"Reshaped Anomaly Scores Shape: {reshaped_anomaly_scores.shape}")

        if reshaped_point_cloud.shape[0] != reshaped_anomaly_scores.shape[0]:
            raise ValueError("Mismatch between reshaped point cloud points and anomaly scores")

        # Print a few samples for understanding
        print("Sample Point Cloud Data and Anomaly Scores:")
        for i in range(min(5, reshaped_point_cloud.shape[0])):
            print(f"Point: {reshaped_point_cloud[i]}, Anomaly Score: {reshaped_anomaly_scores[i]}")

        # Normalize anomaly scores for coloring
        normalized_scores = (reshaped_anomaly_scores - reshaped_anomaly_scores.min()) / (reshaped_anomaly_scores.max() - reshaped_anomaly_scores.min())
        print(f"Normalized Scores Shape: {normalized_scores.shape}")

        # Apply colors based on normalized scores
        colors = plt.cm.jet(normalized_scores)[:, :3]  # (num_points, 3)
        print(f"Colors Shape: {colors.shape}")

        # Create point cloud with Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(reshaped_point_cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([pcd])
        break


def reshape_point_cloud(point_cloud):
    # Flatten the point cloud
    return point_cloud.reshape(-1, 3)

def reshape_anomaly_scores(anomaly_scores):
    # Flatten the anomaly scores
    return anomaly_scores.reshape(-1)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    return model

def evaluate_model(anomaly_scores, labels, threshold):
    predictions = (anomaly_scores > threshold).astype(int)
    accuracy = np.mean(predictions == labels)
    return accuracy

def estimate_anomalies(anomaly_scores, threshold):
    predictions = (anomaly_scores > threshold).astype(int)
    anomaly_ratio = np.mean(predictions)
    print(f"Estimated Anomalies Ratio: {anomaly_ratio * 100:.2f}%")
    return anomaly_ratio

def normalize_scores(scores):
    return (scores - scores.min()) / (scores.max() - scores.min())

def evaluate_student_teacher(student, teacher, dataloader, device, save_to_file=True, file_path='anomaly_scores.txt'):
    loss_fn = nn.MSELoss()
    teacher.eval()
    student.eval()
    results = []
    
    with torch.no_grad():
        if save_to_file:
            f = open(file_path, 'w')
        
        for i, data in enumerate(dataloader):
            data = data.to(device)

            teacher_features = teacher(data)
            student_features = student(data)
            scores = torch.norm(teacher_features - student_features, dim=2)

            point_cloud = data.pos.cpu().numpy()
            anomaly_scores = scores.cpu().numpy()

            # Reshape point cloud and anomaly scores
            reshaped_point_cloud = point_cloud.reshape(-1, 3)  # (batch_size * num_points, 3)
            reshaped_anomaly_scores = anomaly_scores.reshape(-1)  # (batch_size * num_points)

            if save_to_file:
                # Write to file
                for pc, score in zip(reshaped_point_cloud, reshaped_anomaly_scores):
                    f.write(f"{pc[0]}, {pc[1]}, {pc[2]}, {score}\n")
            
            results.append((reshaped_point_cloud, reshaped_anomaly_scores))
            
            # Print for debugging
            if i % 100 == 0:
                print(f'Batch {i}:')
                print(f'Positions: {point_cloud[:5]}')
                print(f'Anomaly Scores: {anomaly_scores[:5]}')

        if save_to_file:
            f.close()

    return results

if __name__ == "__main__":
    print("Started")
    
    feature_dim = 64
    batch_size = 2
    epochs = 8
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
    decoder_path = 'decoder_model.pth'
    
    initialized_teacher = teacher_net.TeacherNet(feature_dim).to(device)
    initialized_student = student_net.StudentNet(feature_dim).to(device)
    decoder = decoder_net.DecoderNet(feature_dim).to(device)
    
    print("Loading train data...")
    train_dataset = MVTec3D(root='./datasets', split='train', fixed_size=fixed_size)
    print(f'Train dataset size: {len(train_dataset)}')

    print("Loading test data...")
    test_dataset = MVTec3D(root='./datasets', split='test', fixed_size=fixed_size)
    print(f'Test dataset size: {len(test_dataset)}')

    print("Preparing dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if path.exists(teacher_path) and path.exists(student_path) and path.exists(decoder_path):
        print("Loading saved models...")
        initialized_teacher = load_model(initialized_teacher, teacher_path, device)
        initialized_student = load_model(initialized_student, student_path, device)
        decoder = load_model(decoder, decoder_path, device)
    else:
        print("Pretraining teacher...")
        pretrain_teacher(initialized_teacher, decoder, train_loader, device, epochs)
        save_model(initialized_teacher, teacher_path)
        save_model(decoder, decoder_path)
        
        print("Training student...")
        train_student(initialized_student, initialized_teacher, train_loader, device, epochs)
        save_model(initialized_student, student_path)
    
    print("Evaluating student and teacher networks...")
    results = evaluate_student_teacher(initialized_student, initialized_teacher, test_loader, device)
