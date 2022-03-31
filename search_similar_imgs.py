# Please provide the full path to the img you want to search against as follows:
# >>python search_similar_imgs.py **path-to-your-img**

import sys
import searchimagelibrary as sil

input_img_path = sys.argv[1]
input_img = sil.pathsToImages(input_img_path)

color_hist_input = sil.colorHistogramFromPath(input_img)
gabor_vec_input = sil.processGaborVectorFromPath(input_img_path)

match_results = sil.colorPlusTextureMatch(input_img, color_hist_input, num_results=5)