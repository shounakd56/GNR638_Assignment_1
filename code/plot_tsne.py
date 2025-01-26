import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from PIL import Image
from cyvlfeat.sift.dsift import dsift
import cv2

def load_data(data_path, categories):
    image_paths, labels = [], []
    for category in categories:
        category_path = os.path.join(data_path, category)
        category_images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.tif'))]
        image_paths.extend(category_images)
        labels.extend([category] * len(category_images))
    return image_paths, labels

DATA_PATH = 'Images'

CATEGORIES = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]

image_paths, labels = load_data(DATA_PATH, CATEGORIES)
print("Plotting tsne for image SIFT descriptors")
keypoints = []
for path in random.sample(image_paths,100):
    img = np.asarray(Image.open(path),dtype='float32')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Grayscaling the img to make it 2D
    frames, descriptors = dsift(img, step=[5,5], fast=True)        
    keypoints.extend(descriptors)

keypoints = np.array(keypoints)
tsne = TSNE(n_components=2, max_iter=500, random_state=42)
# Dimensionality reduction with t-SNE
y = tsne.fit_transform(keypoints)

plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.7, color='blue')
plt.title("t-SNE Visualization of SIFT Descriptors")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
tsne_plot_path = "results/tsne_keypoints.png"
plt.savefig(tsne_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.show()
print(f"t-SNE plot saved")


train_feats_file_50 = 'pickle_files/features_50_train.pkl'
train_feats_file_300 = 'pickle_files/features_300_train.pkl'

vocab_50 = 'pickle_files/vocab_50.pkl'
vocab_300 = 'pickle_files/vocab_300.pkl'

print("Plotting tsne for image features with Vocab Size = 50")
with open(train_feats_file_50, 'rb') as f:
    train_feats = pickle.load(f)  # Load the existing train features
tsne = TSNE(n_components=2, random_state=42)
# Dimensionality reduction with t-SNE
y = tsne.fit_transform(train_feats)

plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.7, color='blue')
plt.title("t-SNE Visualization of image features with Vocab Size = 50")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
tsne_plot_path = "results/tsne_features_50.png"
plt.savefig(tsne_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.show()
print(f"t-SNE plot saved")

print("Plotting tsne for image features with Vocab Size = 300")
with open(train_feats_file_300, 'rb') as f:
    train_feats = pickle.load(f)  # Load the existing train features
tsne = TSNE(n_components=2, random_state=42)
# Dimensionality reduction with t-SNE
y = tsne.fit_transform(train_feats)

plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.7, color='blue')
plt.title("t-SNE Visualization of image features with Vocab Size = 300")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
tsne_plot_path = "results/tsne_features_300.png"
plt.savefig(tsne_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.show()
print(f"t-SNE plot saved")

print("Plotting tsne for vocab cluster centers with Vocab Size = 50")
with open(vocab_50, 'rb') as f:
    vocab = pickle.load(f)  # Load the existing train features
tsne = TSNE(n_components=2, random_state=42)
# Dimensionality reduction with t-SNE
y = tsne.fit_transform(vocab)

plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.7, color='blue')
plt.title("t-SNE Visualization of vocab cluster centers with Vocab Size = 50")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
tsne_plot_path = "results/tsne_vocab_50.png"
plt.savefig(tsne_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.show()
print(f"t-SNE plot saved")

print("Plotting tsne for vocab cluster centers with Vocab Size = 300")
with open(vocab_300, 'rb') as f:
    vocab = pickle.load(f)  # Load the existing train features
tsne = TSNE(n_components=2, random_state=42)
# Dimensionality reduction with t-SNE
y = tsne.fit_transform(vocab)

plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], s=5, alpha=0.7, color='blue')
plt.title("t-SNE Visualization of vocab cluster centers with Vocab Size = 300")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
tsne_plot_path = "results/tsne_vocab_300.png"
plt.savefig(tsne_plot_path, format='png', dpi=300, bbox_inches='tight')
plt.show()
print(f"t-SNE plot saved")
