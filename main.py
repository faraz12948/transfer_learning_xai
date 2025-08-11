# Alzheimer’s MRI Classification - Local Version with GPU Device Handling

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix

# ========================
# DEVICE CONFIGURATION
# ========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU Available: {gpus}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        DEVICE = "/GPU:0"
    except RuntimeError as e:
        print(e)
else:
    print("[INFO] No GPU found. Using CPU.")
    DEVICE = "/CPU:0"

# ========================
# CONFIGURATION
# ========================
# Change this to your local dataset path
data_path = r"dataset"

labels = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
image_size = 170
valid_extensions = ('.jpg', '.jpeg', '.png')

# ========================
# LOAD DATA
# ========================
X, y = [], []
print("[INFO] Loading images...")
for label in labels:
    folder_path = os.path.join(data_path, label)
    if not os.path.exists(folder_path):
        print(f"[WARNING] Missing folder: {folder_path}")
        continue
    for file in tqdm(os.listdir(folder_path), desc=f"Loading {label}"):
        if file.lower().endswith(valid_extensions):
            img = cv2.imread(os.path.join(folder_path, file))
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            y.append(label)

X = np.array(X)
y = np.array(y)
print(f"[INFO] Loaded {len(X)} images.")

# ========================
# VISUALIZE SAMPLE IMAGES
# ========================
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
plt.show()

# ========================
# PREPARE DATA
# ========================
X, y = shuffle(X, y, random_state=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

del X
del y

unique_labels = np.unique(y_train)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
y_train = to_categorical([label_to_index[i] for i in y_train], num_classes=len(unique_labels))
y_test = to_categorical([label_to_index[i] for i in y_test], num_classes=len(unique_labels))

# ========================
# MODEL CONFIGS
# ========================
models_dict = {
    "DenseNet121": tf.keras.applications.DenseNet121,
    "EfficientNetB0": tf.keras.applications.EfficientNetB0
}

configs = {
    "EfficientNet_Config": {
        "model_name": "EfficientNetB0",
        "lr": 0.001,
        "batch_size": 64,
        "dropout_rate": 0.5,
        "dense_units": 1536,
        "optimizer": "adam",
        "trainable_layers": "last_30"
    },
    "DenseNet_Config": {
        "model_name": "DenseNet121",
        "lr": 1e-4,
        "batch_size": 64,
        "dropout_rate": 0.4,
        "dense_units": 1024,
        "optimizer": "adam",
        "trainable_layers": "all"
    }
}

# ========================
# TRAIN & EVALUATE
# ========================
for name, config in configs.items():
    print(f"\n{'='*20}\nRunning config: {name}\n{'='*20}, {config}")

    BaseModelClass = models_dict[config['model_name']]
    base_model = BaseModelClass(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    # Set trainable layers
    if config['trainable_layers'] == "none":
        base_model.trainable = False
    elif config['trainable_layers'].startswith("last_"):
        n = int(config['trainable_layers'].split("_")[1])
        for layer in base_model.layers[:-n]:
            layer.trainable = False
    else:
        base_model.trainable = True

    # Build custom head
    model = base_model.output
    model = GlobalAveragePooling2D()(model)
    model = Dense(config["dense_units"], activation=None)(model)
    model = BatchNormalization()(model)
    model = tf.keras.layers.Activation('relu')(model)
    model = Dropout(rate=config["dropout_rate"])(model)
    model = Dense(len(unique_labels), activation='softmax')(model)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=model)

    # Compile
    optimizer = {
        "adam": tf.keras.optimizers.Adam(learning_rate=config['lr']),
        "sgd": tf.keras.optimizers.SGD(learning_rate=config['lr']),
        "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=config['lr'])
    }[config['optimizer']]
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train inside device scope
    with tf.device(DEVICE):
        start_time = time.time()
        history = model.fit(X_train, y_train, validation_split=0.1, epochs=12,
                            batch_size=config['batch_size'], verbose=1)
        print(f"[INFO] Training time for {name}: {time.time() - start_time:.2f} seconds")

    # Evaluate
    with tf.device(DEVICE):
        y_true_test = np.argmax(y_test, axis=1)
        y_pred_test = np.argmax(model.predict(X_test), axis=1)

    print(f"\nClassification Report for {name}:\n")
    print(classification_report(y_true_test, y_pred_test, target_names=unique_labels))

    cm = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues_r',
                xticklabels=unique_labels, yticklabels=unique_labels)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()
