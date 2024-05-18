from networks import StudentNet, TeacherNet
import torch
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from datasets import load_mvtec_3d_dataset  # Assuming you have a function to load the dataset

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

def pretrain_teacher(teacher, decoder, dataloader, epochs=10):
    optimizer = torch.optim.Adam(list(teacher.parameters()) + list(decoder.parameters()), lr=0.001)
    loss_fn = nn.MSELoss()
    teacher.train()
    decoder.train()
    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            features = teacher(data)
            reconstructions = decoder(features)
            loss = loss_fn(reconstructions, data.pos)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

def train_student(student, teacher, dataloader, epochs=10):
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    teacher.eval()
    student.train()
    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_features = teacher(data)
            student_features = student(data)
            loss = loss_fn(student_features, teacher_features)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

def compute_anomaly_scores(student, teacher, dataloader):
    teacher.eval()
    student.eval()
    anomaly_scores = []
    for data in dataloader:
        with torch.no_grad():
            teacher_features = teacher(data)
            student_features = student(data)
            scores = torch.norm(teacher_features - student_features, dim=1)
            anomaly_scores.append(scores)
    return anomaly_scores

if __name__ == "__main__":
    print("Started")

    # Parameters
    feature_dim = 64
    batch_size = 16
    epochs = 10

    # Initialize networks
    teacher = TeacherNet(feature_dim)
    student = StudentNet(feature_dim)
    decoder = DecoderNet(feature_dim)

    # Load dataset and prepare DataLoader
    train_dataset = load_mvtec_3d_dataset(train=True)
    test_dataset = load_mvtec_3d_dataset(train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Pretrain the teacher network
    pretrain_teacher(teacher, decoder, train_loader, epochs)

    # Train the student network
    train_student(student, teacher, train_loader, epochs)

    # Compute anomaly scores on the test set
    anomaly_scores = compute_anomaly_scores(student, teacher, test_loader)
    for i, scores in enumerate(anomaly_scores):
        print(f'Anomaly scores for sample {i}: {scores}')
