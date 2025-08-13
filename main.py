import os
import time
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms

# ===============================
# Device setup
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ===============================
# Data loading
# ===============================
data_path = "./dataset/OriginalDataset"  # Change this
labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
valid_extensions = ('.jpg', '.jpeg', '.png')
image_size = 170

X = []
y = []

for label in labels:
    folderPath = os.path.join(data_path, label)
    for file in tqdm(os.listdir(folderPath), desc=f"Loading {label}"):
        if file.lower().endswith(valid_extensions):
            img = cv2.imread(os.path.join(folderPath, file))
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Preview images
# ===============================
# Additions for saving/loading models and figures
# ===============================
model_dir = "saved_models"
fig_dir = "saved_figures"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# Save preview images instead of showing
fig, axes = plt.subplots(len(labels), 4, figsize=(12, len(labels) * 3))
for i, label in enumerate(labels):
    indices = np.where(y == label)[0]
    random_indices = np.random.choice(indices, size=min(4, len(indices)), replace=False)
    for j, idx in enumerate(random_indices):
        ax = axes[i, j]
        ax.imshow(cv2.cvtColor(X[idx], cv2.COLOR_BGR2RGB))
        ax.set_title(label)
        ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "preview_images.png"))
plt.close()

# Shuffle
X, y = shuffle(X, y, random_state=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Label encoding
unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_train_idx = np.array([label_to_index[label] for label in y_train])
y_test_idx = np.array([label_to_index[label] for label in y_test])

# ===============================
# Dataset class
# ===============================
class AlzheimerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ===============================
# Transformations
# ===============================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Dataloaders
train_dataset = AlzheimerDataset(X_train, y_train_idx, transform=transform)
test_dataset = AlzheimerDataset(X_test, y_test_idx, transform=transform)

# ===============================
# Model configs
# ===============================
configs = {
    # "deepseek_code_7b": {
    #     "model_name": "efficientnet_b0",
    #     "lr": 0.001,
    #     "batch_size": 64,
    #     "dropout_rate": 0.5,
    #     "dense_units": 1536,
    #     "optimizer": "adam",
    #     "trainable_layers": "last_30"
    # },
    "Random_Search": {
        "model_name": "densenet121",
        "lr": 1e-4,
        "batch_size": 64,
        "dropout_rate": 0.4,
        "dense_units": 1024,
        "optimizer": "adam",
        "trainable_layers": "all"
    }
    ,
    # "Bayesian_Opt": {
    #     "model_name": "densenet121",
    #     "lr": 0.0001,
    #     "batch_size": 32,
    #     "dropout_rate": 0.5,
    #     "dense_units": 1536,
    #     "optimizer": "adam",
    #     "trainable_layers": "all"
    # }
}

# Model dict
model_dict = {
    "densenet121": models.densenet121,
    "efficientnet_b0": models.efficientnet_b0
}

# ===============================
# Training loop
# ===============================
for name, config in configs.items():
    print(f"\n{'='*20}\nRunning config: {name}\n{'='*20}")
    model_path = os.path.join(model_dir, f"{name}.pth")

    # Load base model
    base_model = model_dict[config["model_name"]](weights="IMAGENET1K_V1")
    # Get number of input features for the final layer
    if hasattr(base_model, "classifier"):
        if isinstance(base_model.classifier, nn.Sequential):
            num_features = base_model.classifier[-1].in_features
        else:
            num_features = base_model.classifier.in_features
    elif hasattr(base_model, "fc"):
        num_features = base_model.fc.in_features
    else:
        raise AttributeError("Base model has no classifier or fc attribute.")
    # Freeze layers if needed
    if config["trainable_layers"] == "none":
        for param in base_model.parameters():
            param.requires_grad = False
    elif config["trainable_layers"].startswith("last_"):
        freeze_until = int(config["trainable_layers"].split("_")[1])
        child_counter = 0
        for child in base_model.children():
            child_counter += 1
            if child_counter < (len(list(base_model.children())) - freeze_until):
                for param in child.parameters():
                    param.requires_grad = False
    # "all" means train everything

    # Replace classifier
    base_model.classifier = nn.Sequential(
        nn.Linear(num_features, config["dense_units"]),
        nn.BatchNorm1d(config["dense_units"]),
        nn.ReLU(),
        nn.Dropout(config["dropout_rate"]),
        nn.Linear(config["dense_units"], len(unique_labels))
    )
    base_model = base_model.to(DEVICE)
    # Optimizer
    optimizer = optim.Adam(base_model.parameters(), lr=config["lr"]) if config["optimizer"] == "adam" else optim.SGD(base_model.parameters(), lr=config["lr"], momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # If saved model exists, load it; else train
    if os.path.exists(model_path):
        print(f"[INFO] Loading saved model for {name} from {model_path}")
        base_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"[INFO] Training model for {name}...")
        start_time = time.time()
        base_model.train()
        for epoch in range(12):
            running_loss = 0.0
            correct = 0
            total = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = base_model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * imgs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            print(f"Epoch [{epoch+1}/12] Loss: {running_loss/len(train_loader.dataset):.4f} Acc: {100.*correct/total:.2f}%")

        print(f"Training time for {name}: {time.time() - start_time:.2f} seconds")
        torch.save(base_model.state_dict(), model_path)
        print(f"[INFO] Saved model for {name} to {model_path}")

    # Eval
    base_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = base_model(imgs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=unique_labels))
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"confusion_matrix_{name}.png"))
    plt.close()