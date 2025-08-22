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
from lime import lime_image
from skimage.segmentation import mark_boundaries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F


# ===============================
# Device setup
# ===============================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ===============================
# Data loading
# ===============================
data_path = "./dataset/OriginalDataset"  # Change this
# data_path = "./dataset/NewDataset"  # Change this
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

# ===============================
# Train/val/test split
# ===============================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101, stratify=y_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Label encoding
unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_train_idx = np.array([label_to_index[label] for label in y_train])
y_val_idx   = np.array([label_to_index[label] for label in y_val])
y_test_idx  = np.array([label_to_index[label] for label in y_test])

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
val_dataset   = AlzheimerDataset(X_val, y_val_idx, transform=transform)
test_dataset  = AlzheimerDataset(X_test, y_test_idx, transform=transform)

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
    model_path = os.path.join(model_dir, f"{name}_v2.pth")

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
    val_loader   = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # If saved model exists, load it; else train
    if os.path.exists(model_path):
        print(f"[INFO] Loading saved model for {name} from {model_path}")
        base_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print(f"[INFO] Training model for {name}...")
        start_time = time.time()

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(12):
            base_model.train()
            running_loss, correct, total = 0.0, 0, 0

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

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = 100. * correct / total

            # Validation
            base_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = base_model(imgs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            val_loss /= len(val_loader.dataset)
            val_acc = 100. * val_correct / val_total

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            print(f"Epoch [{epoch+1}/12] "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")

        print(f"Training time for {name}: {time.time() - start_time:.2f} seconds")
        torch.save(base_model.state_dict(), model_path)
        print(f"[INFO] Saved model for {name} to {model_path}")

        # ===== Save curves =====
        plt.figure()
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title(f"Loss Curve - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"loss_curve_{name}.png"))
        plt.close()

        plt.figure()
        plt.plot(history["train_acc"], label="Train Acc")
        plt.plot(history["val_acc"], label="Val Acc")
        plt.title(f"Accuracy Curve - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.savefig(os.path.join(fig_dir, f"accuracy_curve_{name}.png"))
        plt.close()



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
# ===============================
# XAI with LIME
# ===============================
explainer = lime_image.LimeImageExplainer()
# ===============================

def predict_fn(images):
    base_model.eval()
    batch = torch.stack([transform(img) for img in images], dim=0).to(DEVICE)
    with torch.no_grad():
        outputs = base_model(batch)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    return probs

# Pick a few test samples to explain
num_explanations = 5
os.makedirs(os.path.join(fig_dir, "lime"), exist_ok=True)


# ===============================
# Grad-CAM helper
# ===============================
def generate_gradcam(model, img_tensor, target_class):
    gradients = []
    activations = []

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    def forward_hook(module, input, output):
        activations.append(output.detach())

    # Register hooks on the last conv layer
    last_conv_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv_layer = module
    if last_conv_layer is None:
        raise RuntimeError("No Conv2d layer found in model for Grad-CAM")

    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_backward_hook(backward_hook)

    # Forward pass
    output = model(img_tensor)
    model.zero_grad()

    # Backward pass
    class_loss = output[0, target_class]
    class_loss.backward()

    # Get activations & gradients
    grads = gradients[0]
    acts = activations[0]

    pooled_grads = torch.mean(grads, dim=[0, 2, 3])
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    heatmap = torch.mean(acts, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    heatmap /= torch.max(heatmap)
    heatmap = heatmap.cpu().numpy()

    # Clean up hooks
    forward_handle.remove()
    backward_handle.remove()

    return heatmap

# ===============================
# Static file paths for XAI
# ===============================
xai_files = [
    "./dataset/OriginalDataset/MildDemented/27 (10).jpg",
    "./dataset/OriginalDataset/MildDemented/27 (11).jpg",
    "./dataset/OriginalDataset/ModerateDemented/28.jpg",
    "./dataset/OriginalDataset/VeryMildDemented/26 (46).jpg",
    "./dataset/OriginalDataset/NonDemented/26 (64).jpg"
]
# # Static file paths for XAI
# # ===============================
# xai_files = [
#     "./dataset/OriginalDataset/MildDemented/26 (23).jpg",
#     "./dataset/OriginalDataset/MildDemented/27 (10).jpg",
#     "./dataset/OriginalDataset/ModerateDemented/28.jpg",
#     "./dataset/OriginalDataset/VeryMildDemented/26 (46).jpg",
#     "./dataset/OriginalDataset/NonDemented/26 (64).jpg"
# ]

# ===============================
# Run XAI
# ===============================
for idx, file_path in enumerate(xai_files):
    # Load image
    img_np = cv2.imread(file_path)
    img_np = cv2.resize(img_np, (image_size, image_size))
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

    # True class from path
    true_class_name = file_path.split("/")[-2]

    # Predict class
    probs = predict_fn([img_rgb])
    pred_class_idx = np.argmax(probs[0])
    pred_class_name = unique_labels[pred_class_idx]

    # ===== LIME =====
    explanation = explainer.explain_instance(
        img_rgb,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    lime_img = mark_boundaries(temp / 255.0, mask, color=(1, 1, 0))

    # ===== Grad-CAM =====
    img_tensor = transform(img_rgb).unsqueeze(0).to(DEVICE)
    heatmap = generate_gradcam(base_model, img_tensor, pred_class_idx)

    heatmap_resized = cv2.resize(heatmap, (image_size, image_size))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    gradcam_img = cv2.addWeighted(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), 0.6, heatmap_colored, 0.4, 0)

    # ===== Save side-by-side =====
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(lime_img)
    axs[1].set_title("LIME")
    axs[1].axis("off")

    axs[2].imshow(gradcam_img)
    axs[2].set_title("Grad-CAM")
    axs[2].axis("off")

    fig.suptitle(f"True: {true_class_name} | Pred: {pred_class_name}", fontsize=14)
    save_path = os.path.join(fig_dir, "lime", f"{idx}_{true_class_name}_pred_{pred_class_name}_gc.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[INFO] Saved: {save_path}")



