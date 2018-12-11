# coding: utf-8
# author: halloTheCoder

# - No of clusters `k`
# - Pick randomly `k` points
# - Calculate euclidian distance and update the centroid
# - Continue it until no change in centroid location


import argparse
import os
import sys
from shutil import copy2

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', dest = 'k', help = 'no of clusters to be formed', default = 3, type = int)
    parser.add_argument('--input_file', dest = 'input_file', help = 'input file for segmentation', type = str)

    args = parser.parse_args()
    return args

def resize(input_file, output_folder):
    img = Image.open(input_file)
    basewidth = 300
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    img.save(os.path.join(output_folder, 'output.jpg'))
    
    img = Image.open(os.path.join(output_folder, 'output.jpg'))

    return img

def get_vectors(input_file, output_folder):
    img = resize(input_file, output_folder)    

    img_arr = np.asarray(img)
    
    img_height, img_width = img_arr.shape[: 2]
    
    data_vector = np.zeros((img_height * img_width, 5), dtype = np.float32)      # instead of making `data_vector` 2-D array of 5 dim, we take it as 1-D array
                                                                                 # to make it easy to iterate
    pixel_clusters = np.zeros((img_height * img_width), dtype = np.int32)
    
    for xi in range(img_height):
        for yi in range(img_width):
            data_vector[xi * img_width + yi , : 3] = img_arr[xi, yi]
            data_vector[xi * img_width + yi, 3] = xi
            data_vector[xi * img_width + yi, 4] = yi
    
    data_vector_scaled = preprocessing.normalize(data_vector)
    
    return img_arr, data_vector_scaled, pixel_clusters


def get_clusters(img_arr, k, data_vector_scaled, pixel_clusters, output_folder):    
    # set centers
    min_value = np.amin(data_vector_scaled, axis = 0)
    max_value = np.amax(data_vector_scaled, axis = 0)
    
    # print(min_value, max_value)
    
    centers = np.ndarray(shape = (k, 5))
    for idx in range(len(centers)):
        centers[idx] = np.asarray([np.random.uniform(min_value[i], max_value[i]) for i in range(len(min_value))], dtype = np.float32)
    
    iteration = 0
    
    while(True):
        old_centers = centers.copy()
        # set pixels to their clusters
        for idx, data in enumerate(data_vector_scaled):
            pixel_clusters[idx] = np.argmin([euclidean_distances(data_vector_scaled[idx].reshape(1, -1), centers[i].reshape(1, -1)) for i in range(len(centers))])
    
        # check if a cluster is ever empty, if then append a random data point to it.
        cluster_to_check = np.arange(k)
        cluster_empty = np.in1d(cluster_to_check, pixel_clusters)

        for idx, is_cluster in enumerate(cluster_empty):
            if not is_cluster:
                # sets a random pixel to that cluster
                pixel_clusters[np.random.randint(len(pixel_clusters))] = idx

        # Move centers to the centroid of their cluster
        for i in range(k):
            centers[i] = np.mean(data_vector_scaled[np.where(pixel_clusters == i)], axis = 0)
        
        # check for convergence
        print("Centers Iteration num", iteration, ": \n", centers)
        print('Euclidean distance :: ', euclidean_distances(old_centers.reshape(1, -1), centers.reshape(1, -1)))
        if (euclidean_distances(old_centers.reshape(1, -1), centers.reshape(1, -1)) < 1e-5) or iteration > 20:
            break
        
        set_clusters(img_arr, centers, pixel_clusters, iteration, output_folder)
        
        iteration += 1
    
    return centers, pixel_clusters

def set_clusters(img_arr, centers, pixel_clusters, iteration, output_folder):
    img_height, img_width = img_arr.shape[: 2]
    
    img_arr_copy = img_arr.copy()
    
    for idx, pixel_cluster in enumerate(pixel_clusters): 
        xi, yi = idx // img_width, idx % img_width
        
        img_arr_copy[xi, yi] = np.round(centers[pixel_cluster][:3] * 255)
    
    img = Image.fromarray(img_arr_copy)
    img.save(f'{output_folder}/output_{iteration}.jpg')
    return img


if __name__ == '__main__':
    args = parse_args()
    print(args)
    
    input_file = args.input_file
    k = args.k
    
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok = True)
    
    # copy2(input_file, os.path.join(output_folder, 'output.jpg'))

    img_arr, data_vector_scaled, pixel_clusters =  get_vectors(args.input_file, output_folder)
    centers, pixel_clusters = get_clusters(img_arr, k, data_vector_scaled, pixel_clusters, output_folder)
    
    os.system('convert -delay 30 -loop 0 output/*.jpg output/segmentation.gif')