# Please provide the full path to the img you want to add to the index as follows:
# >>python add_img_to_index.py **path-to-your-img**

import sys
import os
import searchimagelibrary as sil

input_img_path = sys.argv[1]
img_to_add_filename = str(input_img_path).split('/')[-1]
img_bank_path = 'sample_images/'

# Calculate color histogram and gabor vector
color_hist_input = sil.colorHistogramFromPath(input_img_path)
gabor_vec_input = sil.processGaborVectorFromPath(input_img_path)

# Move img into catalog
os.replace(input_img_path, img_bank_path + img_to_add_filename)


print(f'Successfully added {img_to_add_filename} to the existing image bank!')
