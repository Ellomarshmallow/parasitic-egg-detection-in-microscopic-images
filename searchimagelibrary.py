
import pandas as pd
import numpy as np
import cv2 as cv
import json

import matplotlib.pyplot as plt
import imutils

# Reading the json as a dict
local_path_root = ""
with open(f"{local_path_root}labels.json") as json_data:
    data = json.load(json_data)
categories_df = pd.DataFrame(data['categories'])
image_labels_df = pd.DataFrame(data['images'])
annotations_df = pd.DataFrame(data['annotations'])

import os
from os import listdir
from os.path import isfile, join

img_feature_df = pd.read_csv(f'{local_path_root}img_library_color_texture_features.csv')

def displayImages(imgs):
  resized_imgs = []
  for i in imgs:
    resized_imgs.append(imutils.resize(i, width = 250))
  numpy_horizontal_concat = np.concatenate(resized_imgs, axis=1)
  return numpy_horizontal_concat

def pathsToImages(paths):
  imgs = []
  for p in paths:
    imgs.append(cv.imread((p)))
  return imgs

def getROI(img):
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  ret, thresh = cv.threshold(gray_img, 10, 255, 0)
  contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

  biggest_contour_index = 0
  for i,c in enumerate(contours):
    if len(c) > len(contours[biggest_contour_index]):
      biggest_contour_index = i
  #bounding box
  x, y, width, height = cv.boundingRect(contours[biggest_contour_index])
  roi_g = gray_img[y:y+height, x:x+width]
  roi_c = img[y:y+height, x:x+width]
  selected_contour = cv.drawContours(gray_img, contours, biggest_contour_index, (145,255,0), 3)

  return roi_c

def colorHistogram(img):
  lower_threshold = 30
  upper_threshold = 220
  hist = cv.calcHist([img], [0, 1, 2], None, [64,64,64], [lower_threshold, upper_threshold, lower_threshold, upper_threshold, lower_threshold, upper_threshold]) #calculate histogram using the image
  hist = cv.normalize(hist, hist).flatten()
  return hist

def colorHistogramFromPath(path):
  img = cv.imread(path)
  # roi_img = getROI(img)
  return colorHistogram(img)

def processGaborVector(img):
  ksize = 3 #5
  sigma = [0.5, 1, 1.5] #[0.5, 1, 1.5, 2]
  theta = [0, np.pi/6, np.pi/3, np.pi/2] #[0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]
  lamda = [np.pi, np.pi/4] #[np.pi, np.pi/4, np.pi/8]
  gamma = 0.5
  phi = 0

  gabor_feature_vec = []
  gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  for s in sigma:
    for t in theta:
      for l in lamda:
        kernel = cv.getGaborKernel((ksize, ksize), s, t, l, gamma, phi, ktype = cv.CV_32F)
        fimg = cv.filter2D(gray_img, cv.CV_8UC3, kernel)
        kernel_resized = cv.resize(kernel, (400,400))
        fimg = cv.filter2D(gray_img, cv.CV_8UC3, kernel)
        mean = fimg.mean()
        std = fimg.std()
        gabor_feature_vec.append(mean)
        gabor_feature_vec.append(std)
  return gabor_feature_vec

def processGaborVectorFromPath(path):
  img = cv.imread(path)
  img_roi = getROI(img)
  return processGaborVector(img_roi)

def addLocalPath(file_name):
  return f"{local_path_root}sample_images/{file_name}"

def addDiseaseName(file_name):
  pos = file_name.find("_")
  return file_name[:pos]

def displayImages(imgs):
  resized_imgs = []
  for i in imgs:
    resized_imgs.append(imutils.resize(i, width = 250))
  numpy_horizontal_concat = np.concatenate(resized_imgs, axis=1)
  return numpy_horizontal_concat

def pathsToImages(paths):
  imgs = []
  for p in paths:
    imgs.append(cv.imread((p)))
  return imgs

def show_images(images, img_titles = [], title = "", figsize=(20,5), columns = 1):
  fig = plt.figure(figsize=figsize)
  fig.suptitle(title) # or plt.suptitle('Main title')
  plt.title = "hello"
  for i, image in enumerate(images):
      plt.subplot(len(images) / columns + 1, columns, i + 1)
      plt.title = "Hello"
      plt.axis('off')
      plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))

def colorMatch(img, df, num_results = 10):
  OPENCV_METHODS = (
	("Correlation", cv.HISTCMP_CORREL),
	("Chi-Squared", cv.HISTCMP_CHISQR),
	("Intersection", cv.HISTCMP_INTERSECT),
	("Hellinger", cv.HISTCMP_BHATTACHARYYA))

  lower_threshold = 30
  upper_threshold = 220
  img_hist = cv.calcHist([img], [0, 1, 2], None, [64,64,64], [lower_threshold, upper_threshold, lower_threshold, upper_threshold, lower_threshold, upper_threshold])
  img_hist = cv.normalize(img_hist, img_hist).flatten()

# loop over the comparison methods
  correl_dist = []
  chisq_dist = []
  intersect_dist = []
  hellinger_dist = []
  for (methodName, method) in OPENCV_METHODS:
    for hist in df['color_histogram']:
      print(len(hist), type(hist))
      print(len(hist.split(" ")))
      print(hist)
      hist = json.loads(hist)
      d = cv.compareHist(img_hist, hist, method)
      if methodName == "Correlation":
        correl_dist.append(d)
      elif methodName == "Chi-Squared":
        chisq_dist.append(d)
      elif methodName == "Intersection":
        intersect_dist.append(d)
      else:
        hellinger_dist.append(d)

  df_copy = df.copy()
  df_copy['correlation'] = correl_dist
  df_copy['chi-square'] = chisq_dist
  df_copy['intersection'] = intersect_dist
  df_copy['hellinger'] = hellinger_dist

  result = {}
  result['correlation'] = df_copy.sort_values(by = 'correlation', ascending = False).head(num_results + 1).tail(-1)
  result['chi-square'] = df_copy.sort_values(by = 'chi-square', ascending = True).head(num_results + 1).tail(-1)
  result['intersection'] = df_copy.sort_values(by = 'intersection', ascending = False).head(num_results + 1).tail(-1)
  result['hellinger'] = df_copy.sort_values(by = 'hellinger', ascending = True).head(num_results + 1).tail(-1)
  return result

def textureMatch(img, df, num_results = 10):
  img_roi = getROI(img)
  img_gbf = processGaborVector(img_roi)
  distance_list = []
  for gbf in df['gabor_features']:
    gbf = json.loads(gbf)
    distance = np.linalg.norm(np.array(img_gbf) - np.array(gbf)) #calculated euclidian distance
    distance_list.append(distance)

  df_copy = df.copy()
  df_copy['texture_match'] = distance_list
  
  result = df_copy.sort_values(by = 'texture_match', ascending = False).head(num_results+1).tail(-1)
  return result

def allMatchResults(img, df, num_results = 10):
  results = colorMatch(img, df)
  results['texture_match'] = textureMatch(img, df)
  return results

def colorPlusTextureMatch(img, df, num_results = 10):
  color_results = colorMatch(img, df, num_results = len(df))['hellinger']
  texture_results = textureMatch(img, df, num_results = len(df))
  merged_results = texture_results.merge(color_results[['id', 'hellinger']], left_on = 'id', right_on = 'id')
  merged_results['mean_match'] = .5*(merged_results['texture_match'] + merged_results['hellinger'])
  return merged_results.sort_values(by = 'mean_match', ascending = True).head(num_results + 1).tail(-1)

#print(img_feature_df.iloc[666]['path'])
#test_img = pathsToImages([img_feature_df.iloc[666]['path']])
#parasite = img_feature_df.iloc[666]['parasite']

#test_results = textureMatch(test_img[0], img_feature_df,5)
#texture_result_images = pathsToImages(test_results['path'])

#show_images(test_img, figsize=(8,5), title = f"Search Query - {parasite}")
#show_images(texture_result_images, title = "Correlation", columns = len(texture_result_images))