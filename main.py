import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split

def load_images_from_folder(folder_path):
    images = []
    labels = []
    filenames = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).resize((224, 224))
                img_array = np.array(img)
                images.append(img_array)
                
                label = filename.split('_')[0]  
                labels.append(label)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return np.array(images), np.array(labels), np.array(filenames)

def extract_features(images):
    """Extract HOG and color features from images"""
    features = []
    
    for img in images:
        # Convert to grayscale for HOG
        gray_img = rgb2gray(img)
        
        # Extract HOG features
        hog_features = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualize=False)
        
        # Extract LBP texture features
        lbp_features = local_binary_pattern(gray_img, P=8, R=1).flatten()
        lbp_hist, _ = np.histogram(lbp_features, bins=10, range=(0, 10))
        
        # Color features: average RGB values
        color_features = np.mean(img, axis=(0, 1))
        
        # Color histogram
        r_hist, _ = np.histogram(img[:,:,0], bins=32, range=(0, 256))
        g_hist, _ = np.histogram(img[:,:,1], bins=32, range=(0, 256))
        b_hist, _ = np.histogram(img[:,:,2], bins=32, range=(0, 256))
        
        # Combine all features
        img_features = np.concatenate([
            hog_features, 
            lbp_hist / np.sum(lbp_hist),  # Normalized LBP histogram
            color_features,
            r_hist / np.sum(r_hist),      # Normalized color histograms
            g_hist / np.sum(g_hist),
            b_hist / np.sum(b_hist)
        ])
        
        features.append(img_features)
    
    return np.array(features)

# Load images from folder
print("Loading image data...")
## Train cleaned_plates
train_cimages, train_clabels, _ = load_images_from_folder('plates/train/cleaned')
## Train dirty plates
train_dimages, train_dlabels, _ = load_images_from_folder('plates/train/dirty')
## Test plates
test_images, test_labels, test_filenames = load_images_from_folder('plates/test')

# Print the shape of the loaded images and labels
print(f"Train cleaned images shape: {train_cimages.shape}")
print(f"Train cleaned labels shape: {train_clabels.shape}")
print(f"Train dirty images shape: {train_dimages.shape}")
print(f"Train dirty labels shape: {train_dlabels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Combine clean and dirty training data
train_images = np.vstack([train_cimages, train_dimages])
# Create binary labels: 0 for clean, 1 for dirty
train_labels = np.array([0] * len(train_clabels) + [1] * len(train_dlabels))

# Convert to float32 and normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Extract better features instead of just flattening
print("Extracting features...")
train_features = extract_features(train_images)
test_features = extract_features(test_images)

print(f"Feature vector shape: {train_features.shape}")

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    train_features, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)

# Define the model
best_model = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=5, min_samples_leaf=1)

# Retrain the best model on all training data
best_model.fit(train_features, train_labels)

# Make predictions on test data
test_preds = best_model.predict(test_features)

# Create submission dataframe
submission = pd.DataFrame({
    'id': [filename.split('.')[0] for filename in test_filenames],
    'label': ['dirty' if pred == 1 else 'clean' for pred in test_preds]
})

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")