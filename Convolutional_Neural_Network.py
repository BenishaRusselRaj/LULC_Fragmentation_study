import rasterio
import os
import time
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder 
from tensorflow.keras.utils import to_categorical
from matplotlib.colors import ListedColormap

#%%
start = time.time()

#%%
stacked_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2015_Merged_clipped_scene_SCP_RT_815.tif"
ground_truth_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2015_Landcover_Mudumalai_NP.tif"

# stacked_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2025_Landsat_merged_RT.tif"
# ground_truth_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2025_Landcover_classified.tif"

name = stacked_file.rsplit('\\', 1)[1].rsplit('.', 1)[0]
output_files_path = stacked_file.rsplit('\\', 1)[0] + '\\'+ name + '_Outputs'
if not os.path.exists(output_files_path):
    os.makedirs(output_files_path)

with rasterio.open(stacked_file) as src:
    stacked_image = src.read()
    profile = src.profile

with rasterio.open(ground_truth_file) as src:
    ground_truth = src.read(1)

stacked_shape = stacked_image.shape
print(f"Original stacked image shape: {stacked_shape}")


reshaped_image = stacked_image.reshape(stacked_shape[0], -1).T
print(f"Reshaped image shape for PCA: {reshaped_image.shape}")


valid_pixels = np.all(reshaped_image, axis=1) & np.all(np.isfinite(reshaped_image), axis=1)
valid_pixels_filtered = reshaped_image[valid_pixels]
print(f"Valid pixels shape: {valid_pixels_filtered.shape}")


num_classes = 3
pca = PCA(n_components=7)
pca_features_fitted = pca.fit_transform(valid_pixels_filtered)


pca_features = np.zeros((7, stacked_shape[1] * stacked_shape[2]))
pca_features[:, valid_pixels] = pca_features_fitted.T
pca_features = pca_features.reshape(7, stacked_shape[1], stacked_shape[2])


pca_features = np.transpose(pca_features, (1, 2, 0))
print(f"PCA features shape for CNN: {pca_features.shape}")


ground_truth_reshaped = ground_truth.flatten()


valid_mask_2d = np.zeros((stacked_shape[1], stacked_shape[2]), dtype=bool)
valid_mask_2d.flat[valid_pixels] = True


x_train = pca_features[valid_mask_2d]
y_train = ground_truth_reshaped[valid_pixels]

print(f"Training data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)
y_categorical = to_categorical(y_encoded, num_classes)

print(f"Categorical labels shape: {y_categorical.shape}")


model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(7,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', 
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])

print("Training model...")
history = model.fit(
    x_train, y_categorical,
    epochs=30,
    validation_split=0.3,
    batch_size=32,
    verbose=1
)

#%%
plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy') 
plt.legend()
plt.show()

#%%
predictions = model.predict(x_train)
predicted_classes = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_classes)

#%%

prediction_image = np.zeros((stacked_shape[1], stacked_shape[2]))
prediction_image.flat[valid_pixels] = predicted_labels

prediction_vis = prediction_image.copy()
prediction_vis[~valid_mask_2d] = -1

unique_classes = np.unique(predicted_labels)
n_classes = len(unique_classes)
colors = plt.cm.tab10(np.linspace(0, 1, n_classes + 1))  # +1 for no-data
cmap = ListedColormap(colors)

plt.figure()
plt.imshow(prediction_vis, cmap=cmap, vmin=-1, vmax=max(unique_classes))
plt.title('CNN Predicted Classification')
plt.axis('off')

#%%
valid_ground_truth = ground_truth.flat[valid_pixels]
accuracy = np.mean(valid_ground_truth == predicted_labels)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%")

#%%
print('--------%s minutes--------' % (round(((time.time() - start) / 60), 2)))