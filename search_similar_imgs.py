# Please provide the full path to the img you want to search against as follows:
# >>python search_similar_imgs.py **path-to-your-img**

import sys
import cv2 as cv
import searchimagelibrary as sil
import pandas as pd
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

input_img_path = sys.argv[1]

local_path_root = ""

def addLocalPath(file_name):
  return f"{local_path_root}sample_images/{file_name}"

img_feature_df = pd.read_csv(f'img_library_color_texture_features.csv')
img_feature_df['path'] = img_feature_df['file_name'].apply(addLocalPath)


input_img = cv.imread(input_img_path)

# Calculate
match_results = sil.colorPlusTextureMatch(input_img, img_feature_df, num_results=5)

# Display search image and results
cv.imwrite("results/query_image.jpg", input_img);
result_images = sil.pathsToImages(match_results['path'])

for i, img in enumerate(result_images):
	cv.imwrite(f"results/Result_{i+1}.jpg", img)

print('Successfully matched images to your query. Please check the >>results<< folder!')
