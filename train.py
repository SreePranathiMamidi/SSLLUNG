import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, recall_score
from config import *
from dataset import Luna16Dataset
from model import DinoClassifier

# Load DINO backbone
backbone = torch.hub.load(
    "facebookresearch/dinov2",
    "dinov2_vitl14"
)

backbone.eval()
backbone.to(DEVICE)

# Dataset
dataset = Luna16Dataset(
    root_dir="path_to_luna16",
    csv_path="path_to_candidates.csv",
    transform=TRANSFORM,
    balance=True
)

train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size

train_set, val_set = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

model = DinoClassifier(backbone).to(DEVICE)

# ------------------------------
# Weighted Loss (important!)
# ------------------------------
weights = torch.tensor([1.0, 5.0]).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LR
)

# ------------------------------
# Training Loop
# ------------------------------
for epoch in range(NUM_EPOCHS):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"\nEpoch {epoch+1}")
    print("Train Loss:", total_loss / len(train_loader))

    # Validation
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)

            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = accuracy_score(targets, preds)
    recall = recall_score(targets, preds)

    print("Validation Accuracy:", acc)
    print("Validation Recall (Sensitivity):", recall)
