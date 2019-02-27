#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:11:06 2019

@author: wangyiming
"""

import os
#from scipy import misc
import matplotlib.image as mpimg 
import numpy as np
import matplotlib.pyplot as plt
import random


"""
Read Train Face images and Nonface images into array
"""

os.getcwd()
#os.chdir("Documents/ncsu course/ncsu 2019 spring/ECE/Project 1")
resolution=10

Train_images=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train")
print(len(Train_images))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train")))#2000 images


Face_Train_images=[name for name in Train_images if name.startswith("face_")==True]
Nonface_Train_images=[name for name in Train_images if name.startswith("nonface_")==True]


print(len(Face_Train_images))
print(len(Nonface_Train_images))


Face_Train_images_arr=np.zeros((1000,resolution,resolution,3))
for i in range(1000):# 3 doesn'r work
    #print(i)
    #type(i)
    rbg_arr=mpimg.imread("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train/"+Face_Train_images[i])
    
    if rbg_arr.shape!=(resolution,resolution,3):
        print("False")
    Face_Train_images_arr[i,:,:,:]=rbg_arr
    
    
Nonface_Train_images_arr=np.zeros((1000,resolution,resolution,3))
for i in range(1000):# 3 doesn'r work
    #print(i)
    #print(Nonface_Train_images[i])
    #type(i)
    rbg_arr=mpimg.imread("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train/"+Nonface_Train_images[i])
    
    if rbg_arr.shape!=(resolution,resolution,3):
        print("False")
    Nonface_Train_images_arr[i,:,:,:]=rbg_arr
   

Face_Train_images=Face_Train_images_arr.reshape((1000,resolution*resolution*3))
Nonface_Train_images=Nonface_Train_images_arr.reshape((1000,resolution*resolution*3))
Train_images=np.zeros((2000,resolution*resolution*3))
Train_images[0:1000,]=Face_Train_images
Train_images[1000:2000,]=Nonface_Train_images


"""
Read Test Face images and Nonface images into array
"""

Test_images=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test")
print(len(Test_images))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test")))#2000 images


Face_Test_images=[name for name in Test_images if name.startswith("face_")==True]
Nonface_Test_images=[name for name in Test_images if name.startswith("nonface_")==True]


print(len(Face_Test_images))
print(len(Nonface_Test_images))


Face_Test_images_arr=np.zeros((100,resolution,resolution,3))
for i in range(100):# 3 doesn'r work
    #print(i)
    #type(i)
    rbg_arr=mpimg.imread("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test/"+Face_Test_images[i])
    
    if rbg_arr.shape!=(resolution,resolution,3):
        print("False")
    Face_Test_images_arr[i,:,:,:]=rbg_arr
    
    
Nonface_Test_images_arr=np.zeros((100,resolution,resolution,3))
for i in range(100):# 3 doesn'r work
    #print(i)
    #print(Nonface_Test_images[i])
    #type(i)
    rbg_arr=mpimg.imread("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test/"+Nonface_Test_images[i])
    
    if rbg_arr.shape!=(resolution,resolution,3):
        print("False")
    Nonface_Test_images_arr[i,:,:,:]=rbg_arr
   

Face_Test_images = Face_Test_images_arr.reshape((100,resolution*resolution*3))
Nonface_Test_images = Nonface_Test_images_arr.reshape((100,resolution*resolution*3))
Test_images = np.zeros((200,resolution*resolution*3))
Test_images[0:100,] = Face_Test_images
Test_images[100:200,] = Nonface_Test_images