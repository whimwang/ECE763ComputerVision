#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:33:51 2019

@author: wangyiming
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 11:21:51 2019

@author: wangyiming
"""

"""
Model 2: mixture of Gaussian
"""


"""
Input_data_orig: RBG images's matrix
Face: true for calculating the probability P(x|face)
K: the number of levels of gaussian
"""


#Function: to estimate parameters of distribution of Input_data
#Return: parameters of distribution
#Para:
#Input_data:Face Train trans or Nonface Train trans

#
"""
scale and center
"""

def EM_Mix_Gaussian(Input_data,K=3): 
    trans=Input_data
    (N,D)=trans.shape
    #start of u and sigma and h
    h_start=np.ones(K)*(1/K)
    u_start=np.ones((K,D))
    sigma_start=np.ones((K,D,D))
    a=list(range(N))
    random.shuffle(a)
    group_size=int(N/K)
    for i in range(K):
        u_start[i,:]=np.mean(trans[a[(group_size*i):(group_size*(i+1))],:],axis=0)
        sigma_start[i,:,:]=np.diag(np.diag(np.cov(trans[a[(group_size*i):(group_size*(i+1))],].transpose())))#D*D
    h_next=np.zeros(K)
    u_next=np.zeros((K,D))
    sigma_next=np.zeros((K,D,D))
    h_current=h_start
    u_current=u_start
    sigma_current=sigma_start
    
    log_x_gaussian=np.ones((N,K))
    S=np.ones((N,K))
    #EM
    for t in range(30):
        #E-step  
        for k in range(K):
            temp_center=trans-u_current[k,:]
            log_x_gaussian[:,k]=-(1/2)*np.sum(np.multiply(np.dot(temp_center,np.linalg.pinv(sigma_current[k,:,:])),temp_center),axis=1)-\
            (1/2)*np.sum(np.log(np.linalg.svd(sigma_current[k,:,:])[1]))
        for k in range(K):
            for n in range(N):
                S[n,k]=h_current[k]/np.sum(h_current*np.exp(log_x_gaussian[n,:]-log_x_gaussian[n,k]))  
        #M-step
        #update u
        u_next=(np.dot(S.transpose(),trans).transpose()/np.sum(S,axis=0)).transpose()#K*D
        #update sigma
        for k in range(K):
           temp_center=trans-u_next[k,:]
           sigma_next[k,:,:]=np.dot(np.multiply(temp_center.transpose(),S[:,k]),temp_center)/np.sum(S[:,k])
           sigma_next[k,:,:]-sigma_current[k,:,:]
        #update h  
        h_next=np.sum(S,axis=0)/np.sum(S)
        
        #check convergence
        delta_u=np.linalg.norm(u_next-u_current)/np.linalg.norm(u_current)
        delta_h=np.linalg.norm(h_next-h_current)/np.linalg.norm(h_current)
        delta_sigma=np.linalg.norm(sigma_next-sigma_current)/np.linalg.norm(sigma_current)
        #print(np.linalg.norm(u_next-u_current))
        #print(delta_u)
        #print(delta_h)
        #print(delta_sigma)
      
        u_current=u_next
        h_current=h_next
        for k in range(K):
           sigma_current[k,:,:]=np.diag(np.diag(sigma_next[k,:,:]))
        
    sigma=sigma_current
    h=h_current
    u=u_current
    return([sigma,h,u])


def Label_Mix_Gaussian(Input_data_orig,K,threshold=0.5):
    N=Input_data_orig.shape[0]
    (Face_sigma,Face_h,Face_u)=EM_Mix_Gaussian(Face_Train_images,K=K)   
    (Nonface_sigma,Nonface_h,Nonface_u)=EM_Mix_Gaussian(Nonface_Train_images,K=K) 
    Face_trans=Input_data_orig#N*D
    Nonface_trans=Input_data_orig
    
    #(x-u)Sigma^(-1)(x-u)
    temp_face=np.zeros((Input_data_orig.shape[0],K))#N*K
    temp_nonface=np.zeros((Input_data_orig.shape[0],K))
    for i in range(K):
        temp_face[:,i]=np.sum(np.multiply(np.dot(Face_trans-Face_u[i,:],np.linalg.pinv(Face_sigma[i,:,:])),Face_trans-Face_u[i,:]),axis=1)
        temp_nonface[:,i]=np.sum(np.multiply(np.dot(Nonface_trans-Nonface_u[i,:],np.linalg.pinv(Nonface_sigma[i,:,:])),Nonface_trans-Nonface_u[i,:]),axis=1)   
    log_det_face=np.zeros(K)
    log_det_nonface=np.zeros(K)
    for i in range(K):
        log_det_face[i]=np.sum(np.log(np.linalg.svd(Face_sigma[i,:,:])[1]))
        log_det_nonface[i]=np.sum(np.log(np.linalg.svd(Nonface_sigma[i,:,:])[1])) 
     
    estimated_label=np.ones(Input_data_orig.shape[0])*2
    
    #no numerical problems
    if(False):
        p_ratio_face_non=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):    
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_nonface-log_det_face[j])-(1/2)*(temp_nonface[n,:]-temp_face[n,j])+np.log(Nonface_h)-np.log(Face_h[j])))
            #print(temp_p)
            p_ratio_face_non[n]=np.sum(1.0/temp_p)

        
        p_ratio_non_face=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_face-log_det_nonface[j])-(1/2)*(temp_face[n,:]-temp_nonface[n,j])*(Face_h/Nonface_h[j])))
            p_ratio_non_face[n]=np.sum(1.0/temp_p)
           
        p_ratio_non_face
        p_ratio_face_non
        useful_index=[i for i in range(Input_data_orig.shape[0]) if p_ratio_non_face[i]!=p_ratio_face_non[i]]
        no_useful_index=[i for i in range(Input_data_orig.shape[0]) if p_ratio_non_face[i]==p_ratio_face_non[i]]
        estimated_label[[i for i in useful_index if p_ratio_non_face[i]!=p_ratio_face_non[i] and p_ratio_face_non[i]>((1-threshold)/threshold)]]=1
        estimated_label[[i for i in useful_index if p_ratio_non_face[i]!=p_ratio_face_non[i] and p_ratio_face_non[i]<=((1-threshold)/threshold)]]=0
        
    #numerical problem exists
    log_p_ratio_face_non=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K): 
            temp_p[j]=np.max(-(1/2)*(log_det_nonface-log_det_face[j])-(1/2)*(temp_nonface[n,:]-\
                  temp_face[n,j])+np.log(Nonface_h)-np.log(Face_h[j]))
        log_p_ratio_face_non[n]=-np.min(temp_p)
    log_p_ratio_non_face=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K):
            temp_p[j]=np.max(-(1/2)*(log_det_face-log_det_nonface[j])-(1/2)*(temp_face[n,:]-\
                  temp_nonface[n,j])+np.log(Face_h)-np.log(Nonface_h[j]))
        log_p_ratio_non_face[n]=-np.min(temp_p)

    
    estimated_label[[i for i in range(N) if log_p_ratio_face_non[i]>-np.log(threshold/(1-threshold))]]=1
    estimated_label[[i for i in range(N) if log_p_ratio_face_non[i]<=-np.log(threshold/(1-threshold))]]=0

    return(estimated_label)
    
  
    
def FR_Mix_Gaussian(Input_data_orig,true_label,K=3,threshold=0.5):
    N=Input_data_orig.shape[0]
    estimated_label=Label_Mix_Gaussian(Input_data_orig,K=K,threshold=threshold)
    ratio_threshold=np.log(threshold/(1-threshold))
    if(isinstance(threshold,np.ndarray)==False):#threshold is a scalar      
        FR=np.zeros(3)
        FR[0]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
        FR[2]=np.mean(np.abs(estimated_label-true_label))
        return(FR)
    

    
def ROC_Mix_Gaussian(Input_data_orig,true_label,ratio_threshold_seq,K=3):
    N=Input_data_orig.shape[0]
    (Face_sigma,Face_h,Face_u)=EM_Mix_Gaussian(Face_Train_images,K=K)   
    (Nonface_sigma,Nonface_h,Nonface_u)=EM_Mix_Gaussian(Nonface_Train_images,K=K) 
    Face_trans=Input_data_orig
    Nonface_trans=Input_data_orig  
    #(x-u)Sigma^(-1)(x-u)
    temp_face=np.zeros((Input_data_orig.shape[0],K))#N*K
    temp_nonface=np.zeros((Input_data_orig.shape[0],K))
    for i in range(K):
        temp_face[:,i]=np.sum(np.multiply(np.dot(Face_trans-Face_u[i,:],np.linalg.pinv(Face_sigma[i,:,:])),Face_trans-Face_u[i,:]),axis=1)
        temp_nonface[:,i]=np.sum(np.multiply(np.dot(Nonface_trans-Nonface_u[i,:],np.linalg.pinv(Nonface_sigma[i,:,:])),Nonface_trans-Nonface_u[i,:]),axis=1)   

    log_det_face=np.zeros(K)
    log_det_nonface=np.zeros(K)
    for i in range(K):
        log_det_face[i]=np.sum(np.log(np.linalg.svd(Face_sigma[i,:,:])[1]))
        log_det_nonface[i]=np.sum(np.log(np.linalg.svd(Nonface_sigma[i,:,:])[1])) 
    if(False):
        #no numerical problems
        p_ratio_face_non=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            temp_p=np.zeros(K)
            for j in range(K):    
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_nonface-log_det_face[j])-(1/2)*(temp_nonface[n,:]-temp_face[n,j])+np.log(Nonface_h)-np.log(Face_h[j])))

            p_ratio_face_non[n]=1/np.sum(temp_p)        
        p_ratio_non_face=np.zeros(Input_data_orig.shape[0])
        for n in range(Input_data_orig.shape[0]):
            #n=0
            temp_p=np.zeros(K)
            for j in range(K):
                temp_p[j]=np.sum(np.exp(-(1/2)*(log_det_face-log_det_nonface[j])-(1/2)*(temp_face[n,:]-temp_nonface[n,j])*(Face_h/Nonface_h[j])))
            p_ratio_non_face[n]=1/np.sum(temp_p)
          
     #numerical problem exists
    log_p_ratio_face_non=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        temp_p=np.zeros(K)
        for j in range(K): 
            #j=0
            temp_p[j]=np.max(-(1/2)*(log_det_nonface-log_det_face[j])-(1/2)*(temp_nonface[n,:]-\
                  temp_face[n,j])+np.log(Nonface_h)-np.log(Face_h[j]))
            #print(temp_p[j])
            
        #print(temp_p)
        
        log_p_ratio_face_non[n]=-np.min(temp_p)
        #print(log_p_ratio_face_non[n])
    #print(log_p_ratio_face_non)
    
    log_p_ratio_non_face=np.zeros(Input_data_orig.shape[0])
    for n in range(Input_data_orig.shape[0]):
        #n=0
        temp_p=np.zeros(K)
        for j in range(K):
            temp_p[j]=np.max(-(1/2)*(log_det_face-log_det_nonface[j])-(1/2)*(temp_face[n,:]-\
                  temp_nonface[n,j])+np.log(Face_h)-np.log(Nonface_h[j]))
        #print(temp_p)
        log_p_ratio_non_face[n]=-np.min(temp_p)
        #print(log_p_ratio_non_face[n])
    #print(log_p_ratio_non_face)
    FR=np.zeros((2,len(ratio_threshold_seq)))
    for i in range(len(ratio_threshold_seq)):
        #face_or_nonface
        ratio_threshold=ratio_threshold_seq[i]
        estimated_label=np.ones(Input_data_orig.shape[0])*2
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if log_p_ratio_face_non[i]>-ratio_threshold]]=1#>-np.log(threshold/(1-threshold))
        estimated_label[[i for i in range(Input_data_orig.shape[0]) if log_p_ratio_face_non[i]<=-ratio_threshold]]=0#<=-np.log(threshold/(1-threshold)
       
        FR[0,i]=np.mean(estimated_label[[i for i in range(N) if true_label[i]==0]])
        FR[1,i]=1-np.mean(estimated_label[[i for i in range(N) if true_label[i]==1]])
    plt.plot(FR[0,:],1-FR[1,:],"r--")
    plt.show()
        

#Evaluate the learned model on the testing images
Test_true_label=np.zeros(200)
Test_true_label[0:100]=1    
Train_true_label=np.zeros(2000)
Train_true_label[0:1000]=1

for K in range(1,7):
    print(K)
    print(FR_Mix_Gaussian(Train_images,Train_true_label,K=K,threshold=0.5))
    
for K in range(1,7):
    print(K)
    print(FR_Mix_Gaussian(Test_images,Test_true_label,K=K,threshold=0.5))
  

(Face_sigma,Face_h,Face_u)=EM_Mix_Gaussian(Face_Train_images,K=5)   
(Nonface_sigma,Nonface_h,Nonface_u)=EM_Mix_Gaussian(Nonface_Train_images,K=5) 


ROC_Mix_Gaussian(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=5)
ROC_Mix_Gaussian(Test_images,true_label=Test_true_label,ratio_threshold_seq=np.arange(-1500,1500,100),K=5)


#mean
(Face_sigma,Face_h,Face_u)=EM_Mix_Gaussian(Face_Train_images,K=K)   
(Nonface_sigma,Nonface_h,Nonface_u)=EM_Mix_Gaussian(Nonface_Train_images,K=K) 

plt.imshow(np.dot(Face_h,Face_u).reshape((10,10,3)).astype(int))
plt.title("mean-Face")

plt.imshow(np.dot(Nonface_h,Nonface_u).reshape((10,10,3)).astype(int))
plt.title("mean-NonFace")


#cov
cov_diag=np.zeros(10*10*3)
for i in range(Face_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(Face_sigma[i,:,:])*Face_h[i]
    
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-Face")

cov_diag=np.zeros(10*10*3)
for i in range(Face_sigma.shape[0]):
    cov_diag=cov_diag+np.diag(Nonface_sigma[i,:,:])*Nonface_h[i]
[min_v,max_v]=[np.min(cov_diag),np.max(cov_diag)]
norm_cov_diag=(cov_diag-min_v)/(max_v-min_v)*255
plt.imshow(norm_cov_diag.reshape((10,10,3)).astype(int))
plt.title("cov-NonFace")