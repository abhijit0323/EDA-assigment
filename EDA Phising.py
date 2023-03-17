#!/usr/bin/env python
# coding: utf-8

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

import os
import requests

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



import json
import time
import pickle
import csv


# In[8]:


URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff"
    
def fetch_data(fname='Training_Dataset.arff'):
    response = requests.get(URL)
    outpath  = os.path.abspath(fname)
    with open(outpath, 'wb') as f:
        f.write(response.content)
    return outpath

data = fetch_data()
print (data)


# In[9]:


def getCSVFromArff(fileNameArff, fileNameSmoted):

    with open(fileNameArff, 'r') as fin:
        data = fin.read().splitlines(True)
    
    i = 0
    cols = []
    for line in data:
        if ('@data' in line):
            i+= 1
            break
        else:
            #print line
            i+= 1
            if (line.startswith('@attribute')):
                if('{' in line):
                    cols.append(line[11:line.index('{')-1])
                else:
                    cols.append(line[11:line.index('numeric')-1])
    
    headers = ",".join(cols)
    
    with open(fileNameSmoted + '.csv', 'w') as fout:
        fout.write(headers)
        fout.write('\n')
        fout.writelines(data[i:])

getCSVFromArff(data, 'Training_Dataset')


# In[10]:


df = pd.read_csv('Training_Dataset.csv')
df.columns = ['having_IP_Address', 'URL_Length', 'Shortining_Service', 'having_At_Symbol', 'double_slash_redirecting', 'Prefix_Suffix', 'having_Sub_Domain', 'SSLfinal_State', 'Domain_registeration_length', 'Favicon', 'port', 'HTTPS_token', 'Request_URL', 'URL_of_Anchor', 'Links_in_tags', 'SFH', 'Submitting_to_email', 'Abnormal_URL', 'Redirect', 'on_mouseover', 'RightClick', 'popUpWidnow', 'Iframe', 'age_of_domain', 'DNSRecord', 'web_traffic', 'Page_Rank', 'Google_Index', 'Links_pointing_to_page', 'Statistical_report', 'Result']
df.to_csv(path_or_buf='Training_Dataset1.txt',header=False, index=False)
df.head()


# In[11]:


sns.countplot(df['URL_Length'],palette='coolwarm',hue='having_IP_Address',data=df)


# In[12]:


sns.countplot(df['having_IP_Address'],palette='coolwarm',hue='double_slash_redirecting',data=df)


# In[13]:


df["having_IP_Address"].value_counts().plot.pie(y=df["having_IP_Address"],figsize=(15,15),autopct='%1.1f%%')


# In[14]:


df["double_slash_redirecting"].value_counts().plot.pie(y=df["double_slash_redirecting"],figsize=(15,15),autopct='%1.1f%%')


# In[15]:


df["Domain_registeration_length"].value_counts().plot.pie(y=df["Domain_registeration_length"],figsize=(15,15),autopct='%1.1f%%')


# In[17]:


dc_cat=df["Links_pointing_to_page"].value_counts()[:10]
category = pd.DataFrame(df['Links_pointing_to_page'].value_counts()) 
plt.figure(figsize=(15,6))
sns.barplot(x=dc_cat, y ='Links_pointing_to_page',data = category[:10],palette='hls')
plt.title('Top 10 Links_pointing_to_page')
plt.xticks(rotation=90)
plt.show()


# In[18]:


dc_cat=df["web_traffic"].value_counts()[:10]
category = pd.DataFrame(df['web_traffic'].value_counts()) 
plt.figure(figsize=(15,6))
sns.barplot(x=dc_cat, y ='web_traffic',data = category[:10],palette='hls')
plt.title('Top 10 web_traffic')
plt.xticks(rotation=90)
plt.show()


# In[19]:


dc_cat=df["Domain_registeration_length"].value_counts()[:10]
category = pd.DataFrame(df['Domain_registeration_length'].value_counts()) 
plt.figure(figsize=(15,6))
sns.barplot(x=dc_cat, y ='Domain_registeration_length',data = category[:10],palette='hls')
plt.title('Top 10 Domain_registeration_length')
plt.xticks(rotation=90)
plt.show()


# In[20]:


df["age_of_domain"].value_counts().plot.pie(y=df["age_of_domain"],figsize=(15,15),autopct='%1.1f%%')


# In[21]:


sns.kdeplot(df['popUpWidnow'])


# In[22]:


sns.kdeplot(df['Domain_registeration_length'])


# In[23]:


sns.kdeplot(df['double_slash_redirecting'])


# In[24]:


sns.kdeplot(df['URL_Length'])


# In[ ]:




