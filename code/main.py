import os
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from build_vocabulary import build_vocabulary
from get_bags_of_sifts import get_bags_of_sifts
from svm_classify import svm_classify
from nearest_neighbor_classify import nearest_neighbor_classify

DATA_PATH = 'Images'

CATEGORIES = [d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))]
# print(CATEGORIES)
CATE2ID = {v: k for k, v in enumerate(CATEGORIES)}
CODEWORDS = [5,10,20,30,40,50,75,100,125,150,200,250,300,400]  # Different codebook sizes for validation

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

def load_data(data_path, categories):
    image_paths, labels = [], []
    for category in categories:
        category_path = os.path.join(data_path, category)
        category_images = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(('.tif'))]
        image_paths.extend(category_images)
        labels.extend([category] * len(category_images))
    return image_paths, labels


def build_confusion_mtx(test_labels_ids, predicted_categories, abbr_categories):
    cm = confusion_matrix(test_labels_ids, predicted_categories)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_normalized, abbr_categories, title='Normalized confusion matrix')
    plt.show()
     
def plot_confusion_matrix(cm, category, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(category))
    plt.xticks(tick_marks, category, rotation=45)
    plt.yticks(tick_marks, category)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plot_file_path = "results/confusion_matrix.png"
    plt.savefig(plot_file_path, format='png', dpi=300, bbox_inches='tight')
def main():
    print("Codewords Array: ",CODEWORDS)
    image_paths, labels = load_data(DATA_PATH, CATEGORIES)

    # Train, Validation, and Test split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=VAL_SPLIT + TEST_SPLIT, stratify=labels, random_state=42) # Stratify for balanced split and Random state for reproducible splits
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT), stratify=temp_labels, random_state=42) # Stratify for balanced split and Random state for reproducible splits
    print("Data Split Completed")

    best_vocab_size = None
    best_val_accuracy = 0
    val_accuracies = []

    for vocab_size in CODEWORDS:
        print(f"Validation for vocab size: {vocab_size}")

        vocab_file_path = f'pickle_files/vocab_{vocab_size}.pkl' # Path to store vocab for this Vocab size as a pickle file

        if os.path.exists(vocab_file_path):
            print(f"Vocabulary file {vocab_file_path} already exists, so skipping vocabulary building")
        else:
            print(f"Vocabulary file {vocab_file_path} not found, so building vocabulary")
            vocab = build_vocabulary(train_paths, vocab_size)
            with open(vocab_file_path, 'wb') as f:
                pickle.dump(vocab, f)
            print(f"Vocabulary saved to {vocab_file_path}.")

        # Paths for storing the train and validation feature pickle files
        train_feats_file = f'pickle_files/features_{vocab_size}_train.pkl'
        val_feats_file = f'pickle_files/features_{vocab_size}_val.pkl'

        # Train features
        if os.path.exists(train_feats_file):
            print(f"Train features file {train_feats_file} already exists, so skipping feature extraction")
            with open(train_feats_file, 'rb') as f:
                train_feats = pickle.load(f)  # Load the existing train features
        else:
            print(f"Train features file {train_feats_file} not found, so extracting features")
            train_feats = get_bags_of_sifts(train_paths, vocab_size) # Extracting SIFt features using the train images and vocab file
            with open(train_feats_file, 'wb') as f:
                pickle.dump(train_feats, f)
            print(f"Train features saved to {train_feats_file}.")

        # Validation features
        if os.path.exists(val_feats_file):
            print(f"Validation features file {val_feats_file} already exists. Skipping feature extraction for validation set.")
            with open(val_feats_file, 'rb') as f:
                val_feats = pickle.load(f)  # Load the existing validation features
        else:
            print(f"Validation features file {val_feats_file} not found. Extracting features for validation set.")
            val_feats = get_bags_of_sifts(val_paths, vocab_size) # Extracting SIFt features using the val images and vocab file
            with open(val_feats_file, 'wb') as f:
                pickle.dump(val_feats, f)
            print(f"Validation features saved to {val_feats_file}.")
        # Training SVM on train data and evaluating on val data
        preds = svm_classify(train_feats, train_labels, val_feats)
        val_accuracy = accuracy_score(val_labels, preds)
        val_accuracies.append(val_accuracy)

        print(f"Validation accuracy for vocab size {vocab_size}: {val_accuracy}")

        # Update best vocab size if val accuracy improves
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_vocab_size = vocab_size

    print(f"Best vocab size: {best_vocab_size} with validation accuracy: {best_val_accuracy}")

    # Plot of val accuracy vs. vocab size
    plt.figure(figsize=(8, 6))
    plt.plot(CODEWORDS, val_accuracies, marker='o', label="Validation Accuracy")
    plt.xlabel("Number of Codewords (Vocabulary Size)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Codewords")
    plt.grid(True)
    plt.legend()
    plot_file_path = "results/accuracy_vs_codewords.png"
    plt.savefig(plot_file_path, format='png', dpi=300, bbox_inches='tight')
    plt.show()

    # Final evaluation on test set using the best vocabulary size
    print("Final Evaluation on test set")
    best_train_feats = f'pickle_files/features_{best_vocab_size}_train.pkl'
    with open(best_train_feats, 'rb') as f:
        train_feats = pickle.load(f)  # Loading the existing train features
    test_feats = get_bags_of_sifts(test_paths, best_vocab_size)
    preds = svm_classify(train_feats, train_labels, test_feats)
    preds2 = nearest_neighbor_classify(train_feats, train_labels, test_feats)
    test_accuracy = accuracy_score(test_labels, preds)
    test_accuracy2 = accuracy_score(test_labels, preds2)

    print(f"Test Accuracy using SVC: {test_accuracy}")
    print(f"Test Accuracy using KNN: {test_accuracy2}")

    print("Class-wise Accuracy:")
    class_report = classification_report(test_labels, preds, target_names=CATEGORIES, digits=4)
    print(class_report)
    test_labels_ids = [CATE2ID[x] for x in test_labels]
    predicted_categories_ids = [CATE2ID[x] for x in preds]
    build_confusion_mtx(test_labels_ids, predicted_categories_ids, CATEGORIES)

    print("\nSummary of Results:")
    print(f"Best Vocabulary Size: {best_vocab_size}")
    print(f"Validation Accuracy (Best): {best_val_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("Accuracy vs. Vocabulary Size:")
    for vocab_size, val_acc in zip(CODEWORDS, val_accuracies):
        print(f"  Codebook Size {vocab_size}: Validation Accuracy = {val_acc:.4f}")


if __name__ == "__main__":
    main()
