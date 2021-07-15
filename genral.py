# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 20:56:48 2020

@author: Hackie_Packie
"""

# Importing the libraries
import numpy as np
import re
import time
from sklearn.metrics import silhouette_score 
from sklearn.metrics import davies_bouldin_score
import scipy
import pandas as pd



lines = open('copyvec1D_new.txt', encoding = 'utf-8', errors = 'ignore').read().split('key')
clust = open('copyvec1DCen_new.txt', encoding = 'utf-8', errors = 'ignore').read().split('key')
# splitting for lines
data = []           
for i in range(1,len(lines)):
    temp = []
    temp =  lines[i].split(';')
    data.append(temp)
# splitting for cluster
cluster = []
for i in range(1,len(clust)):
    temp = []
    temp =  clust[i].split(';')
    cluster.append(temp)

# getting the total no of points
length = int(re.search(r'\d+', data[0][2]).group())


# getting points for data
points=[]
for i in range(0,len(data)):
    temp=[]
    for j in range(3,len(data[i])-1):
        temp.append(data[i][j])
    points.append(temp)


# getting points for cluster
cpoints=[]
for i in range(0,len(cluster)):
    temp=[]
    for j in range(3,len(cluster[i])-1):
        temp.append(cluster[i][j])
    cpoints.append(temp)


# converting to dictionary for data points
vector_data =[]
for i in range(0,len(points)):
    temp2=[]
    for j in range(0,len(points[i])):
        temp=[]
        temp1=[]
        temp = points[i][j].split(':')
        temp1.append(int(temp[0]))
        temp1.append(float(temp[1]))
        temp2.append(temp1)
    vector_data.append(temp2)        

# converting to dictionary for cluster points
centroid_cluster =[]
for i in range(0,len(cpoints)):
    temp2=[]
    for j in range(0,len(cpoints[i])):
        temp=[]
        temp1=[]
        temp = cpoints[i][j].split(':')
        temp1.append(int(temp[0]))
        temp1.append(float(temp[1]))
        temp2.append(temp1)
    centroid_cluster.append(temp2)        

# for vector data
vd=[]

for i in range(0,len(vector_data)):
    temp=[]
    for j in range(0,length):
        temp1=[]
        temp1.append(j)
        temp1.append(0.0)
        temp.append(temp1)
    vd.append(temp)    
              
for i in range(0,len(vector_data)):
    for j in range(0,len(vector_data[i])):        
        k = vector_data[i][j][0]
        vd[i][k][1] = vector_data[i][j][1]

           
# for cluster
cc=[]

for i in range(0,len(centroid_cluster)):
    temp=[]
    for j in range(0,length):
        temp1=[]
        temp1.append(j)
        temp1.append(0.0)
        temp.append(temp1)
    cc.append(temp)    
                   
for i in range(0,len(centroid_cluster)):
    for j in range(0,len(centroid_cluster[i])):        
        k = centroid_cluster[i][j][0]
        cc[i][k][1] = centroid_cluster[i][j][1]
                    



# calculating the distance from each point of vector to each centroid
distance2=[] 

for i in range(0,len(vd)):
    for k in range(0,len(cc)):
        
        sum =0
        for (j,l) in zip(range(0,len(vd[i])),range(0,len(cc[k]))):
            temp1=[]
            x= vd[i][j][1] - cc[k][l][1]
            x = np.square(x)
            sum=sum+x
        
        distance2.append(sum)

distance2 = np.reshape(distance2,(len(vector_data),len(cc))) 
distance1=distance2
    


# arrengingin the given sequence ofcluster if its not in proper order
seq=[]
for i in range(0,len(cluster)):
    val = int(re.search(r'\d+', cluster[i][0]).group())
    seq.append(val)
    
perm_mat = np.zeros((len(seq), len(seq)))

for idx, i in enumerate(seq):
    perm_mat[idx, i] = 1
distance1=  np.dot(distance1, perm_mat)   



# adding the points to the cluster
clusters = []

for i in range(0,len(cc)):
    temp=[]
    clusters.append(temp)
    
        
for j in range(0,len(distance1)):
    min_dist = np.argmin(distance1[j])    
    clusters[min_dist].append(j)    


# assiging the total labels according to the minimum distance of vector points to a cluster which is close
label=np.zeros(len(vd))
for i in range(0,len(clusters)):
    for j in range(0,len(clusters[i])):
        index = clusters[i][j]
        label[index] = i
    

# passing all the vector data
vec =[]
for i in range(0,len(vd)):
    temp = []
    for j in range(0,len(vd[i])):
        temp.append(vd[i][j][1])
    vec.append(temp)    


# calculating the silhouette score
score = silhouette_score (vec, label, metric='euclidean')


    
# silhouette_ score    
sil_score = silhouette_score (vec, label, metric='euclidean')

# davies bouldin score
db_score = davies_bouldin_score(vec, label)


# dunn index

# function defined to calculate the distance
def reshape(lst1, lst2): 
    last = 0
    res = [] 
    for ele in lst1: 
        res.append(lst2[last : last + len(ele)]) 
        last += len(ele) 
          
    return res
inputs  = reshape(clusters,vd)

# getting the final vector points according to the cluster arrengments
per = []
for i in range(0,len(inputs)):
    temp = []
    for j in range(0,len(inputs[i])):
        for k in range(0,len(inputs[i][j])):
            temp.append(inputs[i][j][k])
    per.append(temp)


# defining the function to calculate the distance 
def dis_cal(lst3,lst4):
    length1 = len(lst3)
    diff = 0
    for k in range(0,length1):
        di = lst3[k] - lst4[k]
        di = di**2
        diff = diff + di
    
    return diff
    

# diameter calculation to get max diameter
dia=[]
for i in range(0,len(per)):
    length = len(per[i]) -1
    d = dis_cal(per[0][0], per[0][length])
    dia.append(d)

max_dia = max(dia)

# min distance from each centroid
dis=[]

for i in range(0,len(per)):
    length = len(per[i]) -1
    for j in range(0,len(per)):
        if(per[i] != per[j]):
            d = dis_cal(per[j][0], per[i][length])
            dis.append(d)
dis_min = min(dis)



dunn_index = dis_min / max_dia

            
            




# XB

## getting the index number where the label changes
index = np.where(np.roll(label,1)!=label)[0]

# calculating the upper part of algo

total = 0
clus =[]

for i in range(0,len(clusters)):
    temp = []
    for j in range(0,len(clusters[i])):
        temp.append(distance1[clusters[i][j]][i])
    clus.append(temp)
        
for i in range(0,len(clus)):
    for j in range(0,len(clus[i])):
        total = total + clus[i][j]


# calculating the bottom part of algo
    
cen = []

for i in range(0,len(cc)):
    temp =[]
    for j in range(0,len(cc[i])):
        temp.append(cc[i][j][1])
    cen.append(temp)
        
## finding the min distance bw clusters
cen_distance=[]
for i in range(0,len(cen)-1):
    
    temp = [x1 - x2 for (x1, x2) in zip(cen[i], cen[i+1])]
    temp= [abs(ele) for ele in temp] 
    t= 0
    for j in range(0,len(temp)):
        t = t + temp[j]
    cen_distance.append(t)

min_distance = min(cen_distance)

D = min_distance * (len(vec))

# getting the FSI value

XB_value = total / D

    
    
    
##### FSI VALUE

## uper part is same so we will find second part

temp_mean=[]

for i in range(0,len(cc)):
    temp=[]
    sum_of_cen = 0
    for j in range(0,len(cc[i])):
        sum_of_cen = sum_of_cen +cc[i][j][1]
        temp.append(sum_of_cen)
    temp_mean.append(temp)   


mean=np.zeros(len(temp_mean[0]))

# adding the each centroids points accorfing to their index       
for i in range(0,len(temp_mean)):
    for j in range(0,len(temp_mean[i])):
        
        mean[j] = mean[j] + temp_mean[i][j]   

# calculating the mean from each centroid 
mean = mean/len(cc) 

# calculating  the (ci - mean)^2
ci_mean = []
for i in range(len(temp_mean)):
    add = 0
    for j in range(0,len(temp_mean[i])):
        val = temp_mean[i][j] - mean[j]
        val = val**2
        add = add + val
    ci_mean.append(add)

# adding all ci_mean values
to = 0
for i in range(0,len(ci_mean)):
    to = to + ci_mean[i]

# final formula
FSI_value = total - to


    

