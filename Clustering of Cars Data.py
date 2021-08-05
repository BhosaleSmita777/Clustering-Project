#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


A = pd.read_csv("/users/ranjeetgaikwad/desktop/data science class/Cars93.csv")


# In[3]:


A.head()


# In[4]:


#Domain Knowledge: Customers generally look for price, mileage 


# In[5]:


B = A[["Price", "MPG.city"]]


# In[6]:


from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
trres = km.fit(B)


# In[7]:


trres.labels_


# In[8]:


len(trres.labels_)


# In[9]:


B.shape


# In[58]:


B['cluster_cartype'] = trres.labels_
B['Car_model'] = A.Model


# In[59]:


B.head()


# In[14]:


B.sort_values(by = "cluster_cartype")


# In[17]:


import matplotlib.pyplot as plt
plt.scatter(B.Price, B['MPG.city'], c = B.cluster_cartype)
plt.xlabel("Price")
plt.ylabel("Mileage")


# In[18]:


q = {0:"red", 1:"blue", 2:"black"}


# In[19]:


col = []
for i in B.cluster_cartype:
    col.append(q[i])


# In[20]:


col


# In[21]:


B['color'] = col


# In[22]:


B


# In[23]:


import matplotlib.pyplot as plt
plt.scatter(B.Price, B['MPG.city'], c = B.color)
plt.xlabel("Price")
plt.ylabel("Mileage")


# # Agglomerative Clustering

# In[24]:


C = A[["Price", "MPG.city"]]


# In[25]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
D = pd.DataFrame(ss.fit_transform(C), columns=["Price", "Mileage"])


# In[28]:


from sklearn.cluster import AgglomerativeClustering
agc = AgglomerativeClustering(n_clusters=5, linkage="complete")
res = agc.fit(D)
C.clusters = res.labels_


# In[33]:


import matplotlib.pyplot as plt
plt.scatter(C.Price, C['MPG.city'], c=C.clusters)


# # Dendrogram

# In[55]:


from scipy.spatial import distance_matrix
from scipy.cluster.hierarchy import dendrogram, linkage


# In[56]:


T = pd.DataFrame(distance_matrix(D.values, D.values), index = D.index, columns=C.index)


# In[57]:


dendrogram(linkage(D))
plt.show()


# In[ ]:





# In[ ]:




