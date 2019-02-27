#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 18:07:20 2019

@author: wangyiming
"""

from scipy import special
from scipy.optimize import fsolve
import math

def EM_T(Input_data,v_start=3):
    trans=Input_data
    (N,D)=trans.shape
    #initialize
    u_start=np.mean(trans,axis=0)
    sigma_start=np.cov(trans.transpose())
 
    u_current=u_start
    sigma_current=sigma_start
    v_current=v_start
    for i in range(30):
        #E step
        temp_center_current=trans-u_current
        temp=v_current+np.sum(np.multiply(np.dot(temp_center_current,np.linalg.inv(sigma_current)),temp_center_current),axis=1)
        Exp_h=(v_current+D)/temp#N
        Exp_log_h=special.digamma((v_current+D)/2)-np.log(temp/2)
        
        #M step
        u_next=np.sum(np.multiply(trans.transpose(),Exp_h),axis=1)/np.sum(Exp_h)#D
        temp_center_current=trans-u_next#N*D
        sigma_next=np.dot(np.multiply(temp_center_current.transpose(),Exp_h),temp_center_current)/N
        def f(v):
            return(np.log(v/2)+1-special.digamma(v/2)+np.mean(Exp_log_h-Exp_h))
        v_next=fsolve(f,v_current)
        #check convergence
        delta_u=np.linalg.norm(u_current-u_next)/np.linalg.norm(u_current)
        delta_sigma=np.linalg.norm(sigma_current-sigma_next)/np.linalg.norm(sigma_current)
        delta_v=np.linalg.norm(v_next-v_current)/np.linalg.norm(v_current)
        #print(delta_u)
        #print(delta_sigma)
        #print(delta_v)
        #updatea
        u_current=u_next
        sigma_current=sigma_next
        v_current=v_next

    u=u_current
    sigma=sigma_current
    v=v_current
    return([u,sigma,v])
    


def Log_p_T(Input_data_orig,Face=True,v_start=3):
    (N,D)=Input_data_orig.shape
    if(Face==True):#for Face
        Input_data=Input_data_orig
        [u,sigma,v]=EM_T(Face_Train_images)
    else:#for Nonface
        Input_data=Input_data_orig
        [u,sigma,v]=EM_T(Nonface_Train_images)
        
    temp_center=Input_data-u
    log_p_t_dist=-(1/2)*np.sum(np.log(np.linalg.svd(sigma)[1]))-\
    (v+D)/2*np.log(1+(1/v)*np.sum(np.multiply(np.dot(temp_center,np.linalg.inv(sigma)),temp_center),axis=1))-\
    (D/2)*np.log(math.pi)-(D/2)*np.log(v)-np.log(special.gamma(v/2))+np.log(special.gamma((v+D)/2))
    return(log_p_t_dist)
        
    

def Label_T(Input_data_orig,v_start=3,threshold=0.5):
    delta=Log_p_T(Input_data_orig,Face=True,v_start=v_start)-Log_p_T(Input_data_orig,Face=False,v_start=v_start)#log_p_face-log_p_nonface
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(Input_data_orig.shape[0])
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if delta[i]>ratio_threshold]]=1
    return(estimated_label)
  

def FR_T(Input_data_orig,true_label,v_start=3,threshold=0.5):
    (N,D)=Input_data_orig.shape
    delta=Log_p_T(Input_data_orig,Face=True,v_start=v_start)-Log_p_T(Input_data_orig,Face=False,v_start=v_start)#log_p_face-log_p_nonface
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):#threshold is a scalar
        #face_or_nonface
        estimated_label=np.zeros(N)
        estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
        #False Rate
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return(FR)
 

def ROC_T(Input_data_orig,true_label,ratio_threshold_seq,v_start=3):  
    N=Input_data_orig.shape[0]
    delta=Log_p_T(Input_data_orig,Face=True,v_start=v_start)-Log_p_T(Input_data_orig,Face=False,v_start=v_start)#log_p_face-log_p_nonface
    #ratio_threshold_seq=np.log(threshold_seq/(1-threshold_seq))
    if(isinstance(ratio_threshold_seq,np.ndarray)):#threshold is a seq
        FR=np.zeros((2,len(ratio_threshold_seq)))
        for i in range(len(ratio_threshold_seq)):
            #face_or_nonface
            ratio_threshold=ratio_threshold_seq[i]
            estimated_label=np.zeros(N)
            estimated_label[[i for i in range(N) if delta[i]>ratio_threshold]]=1
            #False Rate
            FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
            FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        plt.plot(FR[0,:],1-FR[1,:],"r--")
        plt.show()             
 
#Evaluate the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1 
Train_true_label=np.zeros(2000)
Train_true_label[0:1000]=1 

print(FR_T(Train_images,true_label=Train_true_label,threshold=0.5,v_start=5))
print(FR_T(Test_images,true_label=Test_true_label,threshold=0.5,v_start=5))
ROC_T(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),v_start=5)

#mean
[Face_u,Face_sigma,Face_v]=EM_T(Face_Train_images,v_start=3)
[Nonface_u,Nonface_sigma,Nonface_v]=EM_T(Nonface_Train_images,v_start=3)

plt.subplot(2, 2, 1)
plt.imshow(Face_u.reshape((10,10,3)).astype(int))
plt.title("mean-Face")

plt.subplot(2, 2, 2)
plt.imshow(Nonface_u.reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")

#cov
plt.subplot(2, 2, 3)
cov_diag=np.diag(Face_sigma)    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")

plt.subplot(2, 2, 4)
cov_diag=np.diag(Nonface_sigma)  
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")