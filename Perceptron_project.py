import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs):
        self.weights = np.zeros(num_inputs + 1)
        self.lr = 0.01

    def predict(self, x):
        x_bias = np.append(x, 1)
        return np.dot(self.weights, x_bias)

    def train(self, X, y, epochs=1000):
        X_bias = np.hstack((X, np.ones((X.shape[0], 1))))
        for epoch in range(epochs):
            updated = False
            for i in range(X.shape[0]):
                activation = np.dot(self.weights, X_bias[i])
                pred = 1 if activation >= 0 else -1
                if pred != y[i]:
                    self.weights += self.lr * y[i] * X_bias[i]
                    updated = True
            if not updated:
                return epoch
        return epochs

def extract_patterns(image_path, tile_h=95, tile_w=95):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    patterns = []
    rows = img_array.shape[0] // tile_h
    cols = img_array.shape[1] // tile_w
    for r in range(rows):
        for c in range(cols):
            tile = img_array[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
            if tile.shape == (95, 95):
                bipolar = np.where(tile < 128, -1, 1).flatten()
                patterns.append(bipolar)
    print(f"Extracted {len(patterns)} patterns")
    return np.array(patterns), img_array 

patterns, train_img_array = extract_patterns('/home/ali/Semnan_Uni/SoftComputing/perceptron-tamrin/Persian char/fars.bmp')

labels = np.repeat(np.arange(17), 7)[:119]

unique_labels = np.unique(labels)
num_classes = len(unique_labels)
print(f"Number of actual classes: {num_classes}")

perceptrons = []
convergence_epochs = []

for cls in unique_labels:
    y = np.where(labels == cls, 1, -1)
    p = Perceptron(9025)
    epochs_needed = p.train(patterns, y)
    convergence_epochs.append(epochs_needed)
    perceptrons.append(p)
    print(f"Class {cls}: Converged after {epochs_needed} epochs")

print("Training done")

plt.figure(figsize=(10,5))
plt.bar(unique_labels, convergence_epochs)
plt.xlabel('Class')
plt.ylabel('Epochs until convergence')
plt.title('Convergence Speed per Class')
plt.grid(axis='y')
plt.show()

counts = np.bincount(labels)
plt.figure(figsize=(10,5))
plt.bar(range(len(counts)), counts)
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Number of Samples per Class')
plt.grid(axis='y')
plt.show()

correct = 0
for i, pat in enumerate(patterns):
    scores = [p.predict(pat) for p in perceptrons]
    pred_idx = np.argmax(scores)
    pred_class = unique_labels[pred_idx]
    if pred_class == labels[i]:
        correct += 1
print(f"Train accuracy: {correct/len(patterns)*100:.2f}%")

def test_new_image(image_path):
    test_img = Image.open(image_path).convert('L')
    test_img = test_img.resize((95, 95))
    test_tile = np.array(test_img)
    bipolar = np.where(test_tile < 128, -1, 1).flatten()
    scores = [p.predict(bipolar) for p in perceptrons]
    pred_idx = np.argmax(scores)
    pred_class = unique_labels[pred_idx]
    
    sample_idx = pred_class * 7
    r = sample_idx // 7
    c = sample_idx % 7
    class_sample = train_img_array[r*95:(r+1)*95, c*95:(c+1)*95]
    
    fig, ax = plt.subplots(1, 2, figsize=(8,4))
    ax[0].imshow(test_tile, cmap='gray')
    ax[0].set_title('Test Image')
    ax[0].axis('off')
    ax[1].imshow(class_sample, cmap='gray')
    ax[1].set_title(f'Predicted Class {pred_class} Sample')
    ax[1].axis('off')
    plt.suptitle(f'Predicted: Class {pred_class}')
    plt.show()
    
    print(f"Predicted class: {pred_class}")
    return pred_class

test_new_image('/home/ali/Semnan_Uni/SoftComputing/perceptron-tamrin/lam.bmp')
