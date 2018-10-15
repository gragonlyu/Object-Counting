
# coding: utf-8

# 1. sift to find descriptors for every image, and stack all the descriptors
# 2. use the stacked descriptor to train a kmean for clustering
# 3. create a histogram for every image
# 4. train a linear model to predict the number count of treelogs

# In[72]:


import os
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
class ObjectCounting():
    def __init__(self,n_cluster):
        self.n_cluster = n_cluster
        

        self.mega_histogram=None
        self.kmean_trained = None
        self.reg = None
    def _get_img(self,folder):       
        for root,_,files in os.walk(folder):
            for file in files:
                file_path = os.path.join(root,file)
                img_list.append(cv2.imread(file_path))
        return img_list, len(img_list)
#     def _img2gray(self,img):
#         return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    def _get_desc(self,img):
        sift = cv2.xfeatures2d.SIFT_create()
        keypoint, descriptor = sift.detectAndCompute(img, None)
        return keypoint, descriptor
    
    def _get_desc_list(self,img_list):
        for img in img_list:
#             gray = self._img2gray(img)
            keypoint, descriptor = self._get_desc(img)
            desc_list.append(descriptor)
        return desc_list
    def _get_stacked_Desc(self,desc_list):
        desc_vstack = np.array(desc_list[0])
        for desc in desc_list[1:]:
            desc_vstack = np.vstack((desc_vstack, desc))
        return desc_vstack
    def _desc_clustering(self,desc_vstack):
        kmean = KMeans(n_clusters = self.n_cluster,verbose=1)
        self.kmean_trained = kmean.fit_predict(desc_vstack)
    def _get_desc_histogram(self,n_images,desc_list):
        self.mega_histogram = np.array([np.zeros(self.n_cluster) for i in range(n_images)])
        jobs = 0
        for img in range(n_images):
            descs = len(desc_list[img])
            for desc in range(descs):
                cluster = self.kmean_trained[jobs+desc]
                self.mega_histogram[img][cluster] += 1
            jobs += descs
    def ModelTraining(self,train,answer):
        train,n_images = self._get_img(train)
        desc_list = self._get_desc_list(train)       
        desc_vstack = self._get_stacked_Desc(desc_list)
        self._desc_clustering(desc_vstack)        
        self._get_desc_histogram(n_images,desc_list)

        X = self.mega_histogram
        Y = np.array(answer['Counts'])
        self.reg = LinearRegression().fit(X, Y)
        
    def predict(self,test):
        test,n_images = self._get_img(test)
        desc_list = self._get_desc_list(test)       
        desc_vstack = self._get_stacked_Desc(desc_list)
        test_clusters = self.kmean_trained.predict(desc_vstack)
        
        mega_histogram = np.array([np.zeros(self.n_cluster) for i in range(n_images)])
        jobs = 0
        for img in range(n_images):
            descs = len(desc_list[img])
            for desc in range(descs):
                cluster = test_clusters[jobs+desc]
                mega_histogram[img][cluster] += 1
            jobs += descs
            
        X = mega_histogram
        predicted_y = self.reg.predict(X)  
        return predicted_y


# In[ ]:


answer=pd.read_excel('C://Users//gvtc4//OneDrive//Desktop//TreeLogs//Image Count.xlsx')
train='C://Users//gvtc4//OneDrive//Desktop//TreeLogs//train'
test='C://Users//gvtc4//OneDrive//Desktop//TreeLogs//test'
ObjectCounting = ObjectCounting(n_cluster=60)
ObjectCounting.ModelTraining(train,answer)


# In[ ]:


predictions = ObjectCounting.predict(test)

