#Please provide the full path to the img you want to add to the index as follows:
#>>python add_img_to_index.py **path-to-your-img**
import sys
import os

img_to_add = sys.argv[1]
img_to_add_filename = str(img_to_add).split('/')[-1]
img_bank_path = 'sample_images/'

os.replace(img_to_add, img_bank_path+img_to_add_filename)
# TODO calculate color and texture features and store new csv