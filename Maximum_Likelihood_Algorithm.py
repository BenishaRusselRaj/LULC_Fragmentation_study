import rasterio
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.metrics import accuracy_score, confusion_matrix, cohen_kappa_score
import time

start = time.time()

#%%
# stacked_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2015_Merged_clipped_scene_SCP_RT_815.tif"
# ground_truth_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2015_Landcover_Mudumalai_NP.tif"

stacked_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2025_Landsat_merged_RT.tif"
ground_truth_file = r"C:\Users\CBEEV\OneDrive - IIT Madras\Thesis Project\Codes\Final\Final\Data\2025_Landcover_classified.tif"

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
reshaped_image = stacked_image.reshape(stacked_shape[0], -1).T

valid_pixels = np.all(reshaped_image, axis=1) & np.all(np.isfinite(reshaped_image), axis=1)
valid_pixels_filtered = reshaped_image[valid_pixels]

pca_features_fitted = PCA(n_components=7).fit_transform(valid_pixels_filtered)

pca_features = pca_features_fitted.T.reshape(7, stacked_shape[1], stacked_shape[2])

pca_shape = pca_features.shape
pca_reshaped = pca_features.reshape(pca_shape[0], -1).T

pca_valid_pixels = pca_reshaped[valid_pixels]

gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=22)
gmm_predictions = gmm.fit_predict(pca_valid_pixels)


classification = np.zeros(pca_reshaped.shape[0])
classification[valid_pixels] = gmm_predictions
classification_image = classification.reshape(pca_shape[1], pca_shape[2])

#%%
plt.figure()
image = plt.imshow(classification_image, cmap='RdYlGn')
plt.title('Maximum Likelihood Classification')
plt.axis('off')
plt.colorbar(image, label='Class (0: None, 1: Vegetation, 2: Barren Land/Others)')
plt.tight_layout()
plt.show()

#%%
vegetation_image = classification_image == 1

plt.figure()
image = plt.imshow(vegetation_image, cmap='Greens')
plt.title('Maximum Likelihood Classification - Vegetation Area')
plt.axis('off')
plt.tight_layout()
plt.show()


#%%
ground_truth_flattened = ground_truth.flatten()
ground_truth_flattened[ground_truth_flattened == 4294967295] = 0
classification_flattened = classification_image.flatten()

valid_mask = (ground_truth_flattened > 0) & (classification_flattened > 0)
ground_truth_valid = ground_truth_flattened[valid_mask]
classification_valid = classification_flattened[valid_mask]

accuracy = accuracy_score(ground_truth_valid, classification_valid)
confusionMatrix = confusion_matrix(ground_truth_valid, classification_valid) 
kappa = cohen_kappa_score(ground_truth_valid, classification_valid)

#%%
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(confusionMatrix)
print(f"Cohen Kappa Coefficient: {kappa:.3f}")

#%% Patch Metrics
vegetation_image_cleaned = ndimage.binary_closing(vegetation_image, structure=np.ones((3, 3))).astype(vegetation_image.dtype)
vegetation_image_cleaned = ndimage.binary_opening(vegetation_image_cleaned, structure=np.ones((3, 3))).astype(vegetation_image_cleaned.dtype)

vegetation_labeled, num_areas = ndimage.label(vegetation_image_cleaned)

print ("Total patches found: %s" % num_areas)
minimum_area_pixels = 100

patches = []
patch_edges = []
for area_label in range(1, num_areas + 1):
    area = (vegetation_labeled == area_label)
    if np.sum(area) >= minimum_area_pixels:
        patches.append(area_label)
        print ("Patch %s: %s pixels" % (area_label, np.sum(area)))
        edge = ndimage.binary_dilation(area) ^ area
        patch_edges.append(edge)

print ("Number of large patches found: %s" % len(patches))

# patch_edges = np.array(patch_edges)
# total_edges = np.sum(patch_edges)
# total_area = stacked_shape[1] * stacked_shape[2]
# edge_density = total_edges / total_area
# edge_density = round(edge_density, 2)
# print ("Edge Density: %s" % edge_density)


output_path = output_files_path + "\\" + name + '_MLA_classification.tif'
with rasterio.open(output_path, 'w', **profile) as output_dataset:
    output_dataset.write(classification_image.astype(rasterio.uint8), 1)

output_path = output_files_path + "\\" + name + '_MLA_classification_vegetation.tif'
with rasterio.open(output_path, 'w', **profile) as output_dataset:
    output_dataset.write(classification_image.astype(rasterio.uint8), 1)


print('--------%s minutes--------' %(round(((time.time()-start) / 60), 2)))