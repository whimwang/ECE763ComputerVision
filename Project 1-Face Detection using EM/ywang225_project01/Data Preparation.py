1#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 14:07:31 2019

@author: wangyiming
"""

"""
Data Preparation
==========================
"""

"""
Read annotation
01  515 faces
02  520 faces
03  516 faces

01  515 non-faces
02  520 non-faces
03  516 non-faces
"""

import os
#import numpy as np
import random
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 
from PIL import Image

os.getcwd()#make sure current directory is Project 1

os.chdir("Documents/ncsu course/ncsu 2019 spring/ECE/Project 1")

resolution=15

index=["01","02","03","04","05"]
face_total_num=0

current_index=index[0]
for current_index in index[0:3]:
    file_annotation=open("FDDB-folds/FDDB-fold-"+current_index+"-ellipseList.txt")
    for line in file_annotation:
        #line=file_annotation.readline()
        image_name = line.rstrip()
        print(image_name)
        image_face_num = int(file_annotation.readline().rstrip())
        print(image_face_num)
        image_file = Image.open("originalPics/"+image_name+".jpg").convert('RGB')
        plt.imshow(image_file)
        #extract face images
        for _ in range(image_face_num):
            temp_range=file_annotation.readline().rstrip().split(" ")
            (range1,range2,angle,x_center,y_center)=([float(i) for i in temp_range[0:5]])
            sq_len=max(range1,range2)
            #print(sq_len)
            area=(x_center-sq_len,y_center-sq_len,x_center+sq_len,y_center+sq_len)
            image_cropped = image_file.crop(area).resize((resolution,resolution),Image.ANTIALIAS)  # size
            plt.imshow(image_cropped)
            #resize is to change to exact size
            #do not use image_cropped.thumbnail(size, Image.ANTIALIAS), it change to max size not exact
            face_total_num=face_total_num+1
            image_cropped.save("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/"+current_index+"/face_"+str(face_total_num)+".jpg")
            #check whether 60,60,60,3
  
        image_file.close()
    file_annotation.close()

        
nonface_total_num=0
for current_index in index[0:3]:
    file_annotation=open("FDDB-folds/FDDB-fold-"+current_index+"-ellipseList.txt")
    for line in file_annotation:
        image_name = line.rstrip()
        print(image_name)
        image_face_num = int(file_annotation.readline().rstrip())
        print(image_face_num)
        image_file = Image.open("originalPics/"+image_name+".jpg") 
        plt.imshow(image_file)
        #extract face images
        for _ in range(image_face_num):
            temp_range=file_annotation.readline()
            x_center=random.uniform(1, image_file.size[0]/4)#margin part are more likely to be background
            y_center=random.uniform(1, image_file.size[1]/4)#
            sq_len=random.uniform(1,min(x_center,y_center))
            image_cropped=image_file.crop((x_center-sq_len,y_center-sq_len,x_center+sq_len,y_center+sq_len)).resize((resolution,resolution),Image.ANTIALIAS) 
            plt.imshow(image_cropped)
            nonface_total_num=nonface_total_num+1
            image_cropped.save("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/"+current_index+"/nonface_"+str(nonface_total_num)+".jpg")
        
        image_file.close()
    file_annotation.close()
        

"""
Put the extracted face images and non-face images seperatively in two all folder.
And randomly select n=1000 as train data and m=100 as test data.
"""

print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All")))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All")))


os.system("pwd")
os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/01/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All")

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/02/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All")

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/03/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All")

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/01/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All")

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/02/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All")

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/03/*.jpg resolution"+ \
          str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All")


face_img=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All")
num_face_img=len(face_img)#1551
nonface_img=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All")
num_nonface_img=len(nonface_img)
print(num_face_img)
print(num_nonface_img)


        
face_train_index = random.sample(range(0,num_face_img),1000)
face_test_index = random.sample([i for i in range(0,num_face_img) if not i in face_train_index],100)
nonface_train_index=random.sample(range(0,num_nonface_img),1000)
nonface_test_index = random.sample([i for i in range(0,num_nonface_img) if not i in nonface_train_index],100)


#face train
for index in face_train_index:
    os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All/face_"+str(index+1)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Train")

#face test
for index in face_test_index:
    os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/All/face_"+str(index+1)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Test")

#nonface train
for index in nonface_train_index:
    os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All/nonface_"+str(index+1)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Train")

#nonface test
for index in nonface_test_index:
    os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/All/nonface_"+str(index+1)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Test")


print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Train")))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Test")))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Train")))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Test")))

os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Train/*.jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train")
os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Train/*.jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train")
os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_face_pics/Test/*.jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test")
os.system("cp resolution"+str(resolution)+"by"+str(resolution)+"/extracted_nonface_pics/Test/*.jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test")

print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train")))
print(len(os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test")))


