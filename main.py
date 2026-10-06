# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import cv2
import numpy as np
import os
import tensorflow as tf
import keras
import csv

## import the handfeature extractor class
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor
from sklearn.metrics.pairwise import cosine_distances

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

train_folder = "traindata"
train_frames_folder = "train_frames"
train_files = sorted(os.listdir(train_folder))

extractor = HandShapeFeatureExtractor.get_instance()
training_vectors = []


count = 0
for file in train_files:
    if file.endswith(".mp4"):    
        video_path = os.path.join(train_folder, file)
        frameExtractor(video_path, train_frames_folder, count)
        count+= 1

   
frame_files = sorted(os.listdir(train_frames_folder))
 
for frames in frame_files:
    image_path = os.path.join(train_frames_folder,frames)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    feature_vector = extractor.extract_feature(image)
    training_vectors.append(feature_vector)
    

# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

# Make sure when i turn into autograder its just "test"
test_folder = os.path.join("Test","TestData")
# test_folder = "test"
test_frames_folder = "test_frames"
test_files = sorted(os.listdir(test_folder))

test_vectors = []

count = 0
for file in test_files:
    if file.endswith(".mp4"):    
        video_path = os.path.join(test_folder, file)
        frameExtractor(video_path, test_frames_folder, count)
        count+=1

test_frame_files = sorted(os.listdir(test_frames_folder))

for frames in test_frame_files:
    image_path = os.path.join(test_frames_folder,frames)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    feature_vector = extractor.extract_feature(image)
    test_vectors.append(feature_vector)



# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

predictions = []
gesture_labels = {
    "Num0": 0,
    "Num1": 1,
    "Num2": 2,
    "Num3": 3,
    "Num4": 4,
    "Num5": 5,
    "Num6": 6,
    "Num7": 7,
    "Num8": 8,
    "Num9": 9,
    "FanDown": 10,
    "FanOff": 11,
    "FanOn": 12,
    "FanUp": 13,
    "LightOff": 14,
    "LightOn": 15,
    "SetThermo": 16
}

training_matrix = np.vstack(training_vectors)

# print(np.argmin(distance))
# print(train_files[36])

for vector in test_vectors:
    
    distance = cosine_distances(vector, training_matrix)
    closest_dist = np.argmin(distance)
    
    gesture_full_name = train_files[closest_dist]
    gesture_name = gesture_full_name.split("_PRACTICE_")[0]
    
    label = gesture_labels[gesture_name]    
    predictions.append(label)
    
# print(predictions)
with open('Results.csv', 'w', newline='') as results:
    wr = csv.writer(results)
    for label in predictions:
        wr.writerow([label])
    


