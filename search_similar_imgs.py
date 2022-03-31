import sys
import searchimagelibrary as sil

print("Please provide the full path to the img you want to search against as follows:")
print(">>python add_img_to_index.py **path-to-your-img**")

input_img = sys.argv[1]

color_hist_input = sil.colorHistogramFromPath(input_img)