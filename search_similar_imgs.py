# Please provide the full path to the img you want to search against as follows:
# >>python search_similar_imgs.py **path-to-your-img**

import sys
import searchimagelibrary as sil
import pandas as pd

input_img_path = sys.argv[1]
input_img = sil.pathsToImages(input_img_path)

# Calculate
color_hist_input = sil.colorHistogramFromPath(input_img)
gabor_vec_input = sil.processGaborVectorFromPath(input_img_path)

match_results = sil.colorPlusTextureMatch(input_img, color_hist_input, num_results=5)

img_feature_df = pd.read_csv(f'img_library_color_texture_features.csv')
parasite = img_feature_df.iloc[666]['parasite']

# Display search image and results
sil.show_images(input_img, figsize=(8, 5), title=f"Search Query - {parasite}")
sil.show_images(match_results['path'], title="Correlation", columns=len(match_results))
