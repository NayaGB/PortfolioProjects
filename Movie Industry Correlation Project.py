#!/usr/bin/env python
# coding: utf-8

# In[32]:


#Netflix Data Cleaning Project

#Import libraries

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.pyplot import figure

get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['figure.figsize']=(12,8) #Adjusts the configuration of the plot we will create

#Importing the data

df = pd.read_csv('/Users/naidanganbold/Desktop/movies.csv')
df.head()


# In[43]:


#Missing Data Check
for col in df.columns:
    missing_pc = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,missing_pc))


# In[47]:


#Replace NA values with 0
df = df.fillna(0)


# In[40]:


#High level view of the Data
df.dtypes


# In[50]:


#Missing Data Check
for col in df.columns:
    missing_pc = np.mean(df[col].isnull())
    print('{} - {}%'.format(col,missing_pc))


# In[59]:


#Reformatting column types

df['budget']=df['budget'].astype('int64')
df['gross']=df['gross'].astype('int64')


# In[60]:


df.head()


# In[64]:


df.sort_values(by=['gross'], inplace=False, ascending=False)


# In[68]:


pd.set_option('display.max_rows', None)


# In[69]:


#Drop any duplicates
df.drop_duplicates()


# In[75]:


#Correlation between variables

#Scatter-plot between budget vs gross
plt.scatter(x=df['budget'],y=df['gross'],c='blue')
plt.title('Move Budget vs Gross Earnings')
plt.xlabel('Gross Earnings')
plt.ylabel('Budget for Film')
plt.show()


# In[82]:


#Plot Budget vs Gross using seaborn (regression)

sns.regplot(x='budget', y='gross', data=df, scatter_kws={'color':'red'}, line_kws={'color':'black'})


# In[83]:


#Numerical correlation between data variables

df.corr()


# In[87]:


correlation_mtrx=df.corr(method='pearson')
sns.heatmap(correlation_mtrx, annot=True)
plt.title('Partial Correlation HeatMap')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[89]:


#Analysis of non-numeric variables in the dataset

#Set all non-numeric variables to unique codes
df_num=df

for col_name in df_num.columns:
    if(df_num[col_name].dtype=='object'):
        df_num[col_name]=df_num[col_name].astype('category')
        df_num[col_name]=df_num[col_name].cat.codes
        
df_num.head()


# In[90]:


correlation_mtrx=df_num.corr(method='pearson')
sns.heatmap(correlation_mtrx, annot=True)
plt.title('Whole Correlation HeatMap')
plt.xlabel('Movie Features')
plt.ylabel('Movie Features')
plt.show()


# In[98]:


#Organize correlation with highest importance

correlation_mtrx = df_num.corr()
corr_pairs = correlation_mtrx.unstack()
sorted_pairs = corr_pairs.sort_values()
high_corr = sorted_pairs[(sorted_pairs) > 0.45]
high_corr


# In[ ]:


#Higher budget gave higher gross
#Higher votes gave higher gross

