# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 10:55:44 2021

@author: shambhu
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import cv2
from collections import Counter

image1 = cv2.imread('car.jfif')
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image1)

# function to convert RGB into hexa format
def RGB2_HEXA(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (700, 500), interpolation = cv2.INTER_AREA)
    
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    classify = KMeans(n_clusters = number_of_colors)
    labels = classify.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = classify.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2_HEXA(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors, shadow = True, autopct='%1.1f%%')

    return rgb_colors

colors(image('car.jfif'), 10, True)
