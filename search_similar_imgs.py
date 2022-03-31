#Please provide the full path to the img you want to search against as follows:
#>>python add_img_to_index.py **path-to-your-img**

import sys
import searchimagelibrary as sil

input_img = sys.argv[1]

color_hist_input = sil.colorHistogramFromPath(input_img)
color_match = sil.colorMatch(input_img, color_hist_input)

#TODO calculate texture stuff and 