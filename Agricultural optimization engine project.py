#!/usr/bin/env python
# coding: utf-8

# ### Agricultural production optimization engine 
# 
# Problem statement- Build a predictive model so as to suggest the most suitable crops to grow based on the available climatic and soil conditions. 
# 
# Goal: achieve precision farming by optimizing the agricultural production. 

# -- This project is intended on Precision Farming 
# 
# - we have to optimize productivity
# - By understanding requirements of climatic and soil conditions for corps 
# - Helps us to cope with weather unpredictibility. 

# In[1]:


## Installing basic libraries

# for manipulations
import numpy as np 
import pandas as pd 

# for data visualization
import matplotlib.pyplot as plt 
import seaborn as sns

# for interactivity
from ipywidgets import interact


# In[2]:


## Importing the dataset 

data = pd.read_csv('data.csv')


# In[4]:


# Lets check the dimensions of dataset 

print ('Dimension of the dataset is:', data.shape)


# In[5]:


# Let's check the head of the dataset 

data.head()


# In[7]:


## Check the missing values in dataset attributewise

data.isnull().sum()


# - Fill NA function is used to replace these missing values with statistical values such as mean, median or mode. 
# 
# - NA means not available 
# 
# - Pandas have functions like fill-NA, drop NA to treat missing values 

# In[9]:


# Lets check the crops present in the dataset 

data['label'].value_counts()


# In[12]:


# Lets check the summary of all crops 

print ('Average Ratio of Nitrogen in the soil is {0:.2f}'.format (data['N'].mean()))
print ('Average Ratio of Phosphorous in the soil is {0:.2f}'.format (data['P'].mean()))
print ('Average Ratio of Potassium in the soil is {0:.2f}'.format (data['K'].mean()))
print ('Average temperature in celcius is {0:.2f}'.format (data['temperature'].mean()))
print ('Average relative humidity in % {0:.2f}'.format (data['humidity'].mean()))
print ('Average PH value of the soil is {0:.2f}'.format (data['ph'].mean()))
print ('Average rainfall in mm is {0:.2f}'.format (data['rainfall'].mean()))


# In[15]:


## Lets check the summary statistics for each of the crops 

@interact 

def summary(crops = list(data['label'].value_counts().index)):
            x = data[data['label']==crops]
            
            print ('.......................')
            print ('Statistics for Nitrogen')
            print ('Minimum Nitrogen required:', x['N'].min())
            print ('Average Nitrogen required:', x['N'].mean())
            print ('Maximum Nitrogen required:', x['N'].max())
            
            print ('.......................')
            print ('Statistics for Phosphorous')
            print ('Minimum Phosphorous required:', x['P'].min())
            print ('Average Phosphorous required:', x['P'].mean())
            print ('Maximum Phosphorous required:', x['P'].max())
            
            print ('.......................')
            print ('Statistics for Potassium')
            print ('Minimum Potassium required:', x['K'].min())
            print ('Average Potassium required:', x['K'].mean())
            print ('Maximum Potassium required:', x['K'].max())
            
            
            print ('.......................')
            print ('Statistics for Temperature')
            print ('Minimum Temperature required:', x['temperature'].min())
            print ('Average Temperature required:', x['temperature'].mean())
            print ('Maximum Temperature required:', x['temperature'].max())
            
            print ('.......................')
            print ('Statistics for Humidity')
            print ('Minimum Humidity required:', x['humidity'].min())
            print ('Average Humidity required:', x['humidity'].mean())
            print ('Maximum Humidity required:', x['humidity'].max())
            
            print ('.......................')
            print ('Statistics for rainfall')
            print ('Minimum rainfall required:', x['rainfall'].min())
            print ('Average rainfall required:', x['rainfall'].mean())
            print ('Maximum rainfall required:', x['rainfall'].max())


# In[19]:


# Lets make this function more intuitive 

@interact 

def compare (condition = ['N', 'P', 'K', 'temperature', 'ph', 'humidity', 'rainfall']):
    print ('Crops which require greater than average', condition)
    print (data[data[condition]>data[condition].mean()]['label'].unique())
    
    print ('..................................................')
    print ('Crops which require less than average', condition)
    print (data[data[condition]<=data[condition].mean()]['label'].unique())


# In[29]:


## Distribution plot of all attributes
plt.figure(figsize = (15,7))

plt.subplot(2,4,1)
sns.distplot(data['N'], color= 'darkblue')
plt.xlabel ('Ratio of Nitrogen', fontsize =12)
plt.grid()

plt.subplot(2,4,2)
sns.distplot(data['P'], color= 'grey')
plt.xlabel (' Ratio of Phosphorous', fontsize =12)
plt.grid()

plt.subplot(2,4,3)
sns.distplot(data['K'], color= 'red')
plt.xlabel ('Ratio of Potassium', fontsize =12)
plt.grid()

plt.subplot(2,4,4)
sns.distplot(data['temperature'], color= 'lightgrey')
plt.xlabel ('Temperature', fontsize =12)
plt.grid()


plt.subplot(2,4,5)
sns.distplot(data['humidity'], color= 'brown')
plt.xlabel ('Humidity', fontsize =12)
plt.grid()

plt.subplot(2,4,6)
sns.distplot(data['ph'], color= 'darkgreen')
plt.xlabel ('pH level', fontsize =12)
plt.grid()

plt.subplot(2,4,7)
sns.distplot(data['rainfall'], color= 'pink')
plt.xlabel ('rainfall', fontsize =12)
plt.grid()

plt.suptitle('Distribution for agricultural conditions', fontsize = 20)
plt.show()


# In[36]:


## important patterns in dataset 

print('Some interesting patterns discorvered')

print ('Crops which require High ratio of Nitrogen content in soil is:', data[data['N']> 120]['label'].unique())
print ('Crops which require High ratio of Phosphorous content in soil is:', data[data['P']> 100]['label'].unique())
print ('Crops which require High ratio of Potassium  content in soil is:', data[data['K']> 200]['label'].unique())
print ('Crops which require High rainfall:', data[data['rainfall']> 200]['label'].unique())
print ('Crops which require low temperature:', data[data['temperature']< 10]['label'].unique())
print ('Crops which require high temperature:', data[data['temperature']> 40]['label'].unique())
print ('Crops which require low humidity:', data[data['humidity']> 20]['label'].unique())
print ('Crops which require low pH:', data[data['ph']< 4]['label'].unique())
print ('Crops which require high pH:', data[data['ph']> 9]['label'].unique())


# In[59]:


## lets see which crops can be grown in summer, winter and rainy season 

print ('Summer crops:', data[(data['temperature']>30) & (data['humidity']>50)]['label'].unique())
print('.......................................................')
print ('Winter crops:', data[(data['temperature']<20) & (data['humidity']>30)]['label'].unique())
print('........................................................')
print ('Rainy crops:', data[(data['rainfall']>200) & (data['humidity']>30)]['label'].unique())


# In[60]:


# import relevant library

from sklearn.cluster import KMeans 

x = data.drop(['label'], axis = 1)


# In[63]:


x= x.values


# In[65]:


# check the shape 

x.shape


# In[72]:


## elbow plot

wcss = []

for i in range(1,11):
    km = KMeans(n_clusters = i, init = 'k-means++',n_init = 10, random_state = 0)
    km.fit(x)
    wcss.append(km.inertia_)

    ## ploting the graph 

plt.plot(range(1,11), wcss)
plt.title('The elbow methond', fontsize = 20)
plt.xlabel('No.of clusters')
plt.ylabel('WCSS')
plt.show()


# In[87]:


# Lets implement the k means algorithm to perform cluster analysis 

km = KMeans (n_clusters = 4, init ='k-means++', max_iter = 300, n_init = 10, random_state = 0)

y_means = km.fit_predict(x)
y_means = pd.DataFrame(y_means)


# In[90]:


a = data['label']
z = pd.concat([y_means,a], axis = 1)


# In[94]:


z= z.rename(columns = {0:'cluster'})


# In[97]:


# Lets check the cluster of each crop 

print('Crops in first cluster:', z[z['cluster']== 0]['label'].unique())
print ('.....................................')
print('Crops in second cluster:', z[z['cluster']== 1]['label'].unique())
print ('.....................................')
print('Crops in thrid cluster:', z[z['cluster']== 2]['label'].unique())
print ('.....................................')
print('Crops in fourth cluster:', z[z['cluster']== 3]['label'].unique())




# In[101]:


# Lets split the dataset 

y = data['label']
x = data.drop(['label'], axis = 1)

print('Shape of x:', x.shape)
print('Shape of y:', y.shape)


# In[103]:


# training and testing sets

from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split (x,y,test_size = 0.2, random_state = 0)

print('Shape of x train:', x_train.shape)
print('Shape of x test:', x_test.shape)
print('Shape of y train:', y_train.shape)
print('Shape of y test:', y_test.shape)


# In[105]:


## Lets create the predictive model 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit (x_train,y_train)
y_pred = model.predict(x_test)


# In[121]:


# Lets evaluate the model performance

from sklearn.metrics import confusion_matrix

# Lets print the confusion matrix first 
plt.figure(figsize = (20,10))

cm = confusion_matrix (y_test, y_pred)
sns.heatmap(cm, annot = True)
plt.title('Confusion matrix for Logistic regression', fontsize = 15)
plt.show()


# In[120]:


## Lets print the classification report 
from sklearn.metrics import classification_report
cr = classification_report(y_test,y_pred)
print(cr)


# In[123]:


## Lets predict with new data 

data.head(1)


# In[126]:


# prediction of new climatic condition

new_Data = np.array([90,40,40,20,80,7,200])


# In[131]:


prediction = model.predict([new_Data])
print('The suggested crop for this climatic condition is:', prediction)


# In[ ]:




