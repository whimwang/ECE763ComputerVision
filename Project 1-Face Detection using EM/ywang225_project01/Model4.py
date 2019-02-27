#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 21:54:44 2019

@author: wangyiming
"""

"""
factor analysis
"""

import matplotlib.pyplot as plt

def EM_Factor(Input_data,K_sub=3):#K_sub: number of subspace
    trans=Input_data
    (N,D)=trans.shape
    #Initialize
    u=np.mean(trans,axis=0)
    Input_data=Face_Train_images
    sigma_full=np.cov(Face_Train_images.transpose())
    [U_matrix,D_matrix,V_matrix]=np.linalg.svd(sigma_full)
    eta_start=np.multiply(np.sqrt(D_matrix[0:K_sub]),U_matrix[:,0:K_sub])#
    sigma_start=np.diag(np.diag(sigma_full)-np.diag(np.dot(eta_start,eta_start.transpose())))
    
    sigma_current=sigma_start
    eta_current=eta_start
    
    Exp_hh=np.zeros((N,K_sub,K_sub))
    sum_Exp_hh=np.zeros((K_sub,K_sub))
    
    for t in range(60):
        #print(t)
        #E
        sigma_current_inv=np.linalg.pinv(sigma_current)
        a=np.dot(np.dot(eta_current.transpose(),sigma_current_inv),eta_current)
        
        b=np.linalg.pinv(a+np.identity(K_sub))
        d=trans-u
        
        Exp_h=np.dot(np.dot(np.dot(b,eta_current.transpose()),sigma_current_inv),d.transpose()).transpose()#N*K
 
        for i in range(N):
            Exp_hh[i,:,:]= np.outer(Exp_h[i,:],Exp_h[i,:])+b
            sum_Exp_hh=sum_Exp_hh+Exp_hh[i,:,:]
        #M
        e=np.dot(Exp_h.transpose(),d)#K*D
       
        eta_next=np.dot(e.transpose(),np.linalg.pinv(sum_Exp_hh))#D*K
        sigma_next=np.diag(np.diag(np.dot(d.transpose(),d))-np.diag(np.dot(eta_next,e)))/N#D*D
        
        #check convergence
        delta_eta=np.linalg.norm(eta_next-eta_current)/np.linalg.norm(eta_current)
        delta_sigma=np.linalg.norm(sigma_next-sigma_current)/np.linalg.norm(sigma_current)
        #print(delta_eta)
        #print(delta_sigma)
        #update
        eta_current=eta_next
        sigma_current=sigma_next
        
    eta=eta_current
    sigma=sigma_current
    
    return([u,eta,sigma])

  
def Log_p_Factor(Input_data_orig,Face=True,K_sub=3):
    if(Face==True):#for Face
        Input_data=Input_data_orig
        [u,eta,sigma]=EM_Factor(Face_Train_images,K_sub=K_sub)
    else:#for Nonface
        Input_data=Input_data_orig
        [u,eta,sigma]=EM_Factor(Nonface_Train_images,K_sub=K_sub)
        
    temp_1=np.dot(eta,eta.transpose())+sigma
    temp_2=Input_data-u
    log_p=-(1/2)*np.sum(np.log(np.linalg.svd(temp_1)[1]))-(1/2)*np.sum(np.multiply(np.dot(temp_2,np.linalg.pinv(temp_1)),temp_2),axis=1)
    return(log_p)
    
def Label_Factor(Input_data_orig,K_sub=3,threshold=0.5):
    delta=Log_p_Factor(Input_data_orig,Face=True,K_sub=K_sub)-Log_p_Factor(Input_data_orig,Face=False,K_sub=K_sub)#log_p_face-log_p_nonface
    ratio_threshold=np.log(threshold/(1-threshold))
    
    if(isinstance(threshold,np.ndarray)==False):
        estimated_label=np.zeros(Input_data_orig.shape[0])
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if delta[i]>ratio_threshold]]=1
    return(estimated_label)
    
#FR
def FR_Factor(Input_data_orig,true_label,K_sub=3,threshold=0.5):
    N=Input_data_orig.shape[0]
    delta=Log_p_Factor(Input_data_orig,Face=True,K_sub=K_sub)-Log_p_Factor(Input_data_orig,Face=False,K_sub=K_sub)#log_p_face-log_p_nonface
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

#ROC curve
def ROC_Factor(Input_data_orig,true_label,ratio_threshold_seq,K_sub=3):
    N=Input_data_orig.shape[0]
    delta=Log_p_Factor(Input_data_orig,Face=True,K_sub=K_sub)-Log_p_Factor(Input_data_orig,Face=False,K_sub=K_sub)#log_p_face-log_p_nonface
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


print(FR_Factor(Train_images,true_label=Train_true_label,threshold=0.5,K_sub=3))
print(FR_Factor(Test_images,true_label=Test_true_label,threshold=0.5,K_sub=3))
ROC_Factor(Train_images,true_label=Train_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K_sub=3)


[Face_u,Face_eta,Face_sigma]=EM_Factor(Face_Train_images,3)
[Nonface_u,Nonface_eta,Nonface_sigma]=EM_Factor(Nonface_Train_images,3)

#mean
plt.imshow(Face_u.reshape((10,10,3)).astype(int))
plt.title("mean-Face")

plt.imshow(Nonface_u.reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")

#cov
cov_diag=np.diag(np.dot(Face_eta,Face_eta.transpose())+Face_sigma)    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")

cov_diag=np.diag(np.dot(Nonface_eta,Nonface_eta.transpose())+Nonface_sigma) 
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")