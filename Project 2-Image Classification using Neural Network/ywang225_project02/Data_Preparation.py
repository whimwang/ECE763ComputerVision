#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:14:41 2019

@author: wangyiming
"""

"""
Data Preparation
"""

import os
import torch
from PIL import Image
import random
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 





assert(os.getcwd()=="/Users/wangyiming/Documents/ncsu course/ncsu 2019 spring/ECE/Project 2")

#os.chdir("Documents/ncsu course/ncsu 2019 spring/ECE/Project 2")
#os.chdir("..")
#print(os.getcwd())

index=["01","02","03","04","05","06","07","08","09","10"]
resolution=60
face_total_num=0


#5171
current_index=index[0]
for current_index in index[0:10]:
    file_annotation=open("FDDB-folds/FDDB-fold-"+current_index+"-ellipseList.txt")
    for line in file_annotation:
        #line=file_annotation.readline()
        image_name = line.rstrip()
        #print(image_name)
        image_face_num = int(file_annotation.readline().rstrip())
        #print(image_face_num)
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
    print(face_total_num)

        
#print(current_index) 
#print(face_total_num)    
    
nonface_total_num=0
for current_index in index[0:10]:
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
    print(face_total_num)
 
    
    


for i in range(5001,5172):
    os.system("mv resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train/face_"+str(i)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test/face_"+str(i)+".jpg")
    os.system("mv resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train/nonface_"+str(i)+".jpg resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test/nonface_"+str(i)+".jpg")

Train=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Train/")
Test=os.listdir("resolution"+str(resolution)+"by"+str(resolution)+"/extracted_pics/Test/")

num_Train=len(Train)   
num_Test=len(Test) 
print(num_Train)
print(num_Test) 

#len([x for x in Test if not x.endswith(".jpg")])
 
    


