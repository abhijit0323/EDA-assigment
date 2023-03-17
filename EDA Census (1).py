#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
warnings.filterwarnings("ignore")

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('https://raw.githubusercontent.com/abhijit0323/EDA-assigment/main/adult.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


dc=df.copy()


# In[6]:


dc.shape


# In[7]:


dc.head(3)


# In[8]:


dc['Age'].unique()


# In[9]:


dc.info()


# In[10]:


dc.isnull().sum()#no null values


# In[11]:


df.head()


# In[12]:


def chk(data):
    for i in df['Age']:
        if type(i)=='int64':
            print('ok')
        else:
            print(i)
        


# In[ ]:





# In[13]:


dc.info()


# In[14]:


for i in dc.columns:
    num=[]
    obj=[]
    if dc[i].dtype=='int64':
        print(f'{i} is int64')
        num=num.append(dc[i])

    elif dc[i].dtype=='object':
        print(f'{i} is obj')
        obj=obj.append(i)
        
    else:
        print("why here")


# In[15]:


df.head(2)


# In[16]:


dc.info()


# In[ ]:





# In[17]:


dc.columns


# In[18]:


dc.rename(
    columns={"fnlwgt": "final_weight", "education-num": "education_name", "marital-status": "marital_status","capital-gain": "capital_gain","capital-loss": "capital_loss","hours-per-week": "hours_per_eek","native-country": "native_country","Income/year":"Incomeperyear"},
    inplace=True,
)


# In[19]:


dc.info()


# In[20]:


dc[dc.Incomeperyear.str.isnumeric()]


# In[21]:


dc.head()


# In[ ]:





# In[22]:


dc.drop(dc.loc[dc['workclass']==' ?'].index, inplace=True)


# In[23]:


dc.loc[dc['workclass']==' ?']


# In[24]:


dc.info()


# In[25]:


dc['Incomeperyear'].unique()


# In[26]:


dc.drop(dc.loc[dc['native_country']==' ?'].index, inplace=True)


# In[27]:


dc.to_csv("census_cleaned_data.csv",index=False)


# In[28]:


dc.tail()


# In[29]:


dc.head()


# In[30]:


dc.sample(10)


# In[31]:


dc.describe(include='all').T


# In[32]:


dc.head()


# In[33]:


dc.info()


# In[34]:


dc['native_country'].value_counts(normalize=True)


# In[35]:


dc.info()


# In[36]:


plt.figure(figsize=(15,15))
sns.countplot(x=dc['native_country'])


# In[37]:


dc["native_country"].value_counts().plot.pie(y=dc["native_country"],figsize=(10,10),autopct='%1.1f%%')


# In[38]:


sns.kdeplot(dc['Age'])


# In[39]:


sns.countplot(x=df["sex"])


# In[40]:


plt.figure(figsize=(15,15))
sns.countplot(x=df["race"])


# In[41]:


df.info()


# In[42]:


plt.figure(figsize=(15,15))
sns.countplot(x=df["education"])


# In[43]:


# categorical columns
plt.figure(figsize=(20, 15))
plt.suptitle('Univariate Analysis of Categorical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)
category = [ 'workclass', 'education']
for i in range(0, len(category)):
    plt.subplot(2, 2, i+1)
    sns.countplot(x=dc[category[i]],palette="Set2")
    plt.xlabel(category[i])
    plt.xticks(rotation=45)
    plt.tight_layout() 


# In[44]:


df.head()


# In[45]:


numeric_features = [feature for feature in dc.columns if dc[feature].dtype != 'O']


# In[46]:


categorical_features = [feature for feature in dc.columns if dc[feature].dtype == 'O']


# In[47]:


numeric_features


# In[48]:


categorical_features


# In[49]:


sns.kdeplot(df['Age'])


# In[50]:


plt.figure(figsize=(15, 15))
plt.suptitle('Univariate Analysis of Numerical Features', fontsize=20, fontweight='bold', alpha=0.8, y=1.)

for i in range(0, len(numeric_features)):
    plt.subplot(5, 3, i+1)
    sns.kdeplot(x=dc[numeric_features[i]],shade=True, color='r')
    plt.xlabel(numeric_features[i])
    plt.tight_layout()


# In[51]:


dc["race"].value_counts().plot.pie(y=dc["race"],figsize=(15,15),autopct='%1.1f%%')


# In[52]:


dc_cat=dc["native_country"].value_counts()[:10]
category = pd.DataFrame(dc['native_country'].value_counts()) 
plt.figure(figsize=(15,6))
sns.barplot(x=dc_cat, y ='native_country',data = category[:10],palette='hls')
plt.title('Top 10 App Countrywise Workforce')
plt.xticks(rotation=90)
plt.show()


# In[53]:


categorical_features


# In[54]:


numeric_features


# In[55]:


dc.head(1)


# In[56]:


dc_age_capital=dc.groupby(['Age'])['capital_gain'].sum().sort_values(ascending=False).reset_index()


# In[57]:


dc['Age'].unique()


# In[58]:


plt.figure(figsize=(18,18))
sns.barplot(x="Age",y="capital_gain",data=dc_age_capital)


# In[59]:


plt.figure(figsize = (30,30))
sns.set_context("talk")
sns.set_style("darkgrid")

ax = sns.barplot(x = 'Age' , y = 'capital_gain' , data = dc_age_capital )
ax.set_xlabel('Age')
ax.set_ylabel('')
ax.set_title("Max capital gain", size = 20)


# In[60]:


dc_education_capital=dc.groupby(['education'])['capital_gain'].sum().sort_values(ascending=False).reset_index()


# In[61]:


dc_education_capital


# In[62]:


dfa=dc.groupby(['education','native_country'])['capital_gain'].sum().reset_index()


# In[63]:


dfa=dfa.sort_values('capital_gain',ascending=False)


# In[64]:


dfa


# In[65]:


dfa[dfa.education=='Bachelors'][:5]


# In[66]:


dc.head(2)


# In[67]:


pip install jupyterthemes


# In[68]:


get_ipython().system('jt -l')


# In[71]:


get_ipython().system('jt -t monokai')


# In[70]:


pip install --upgrade jupyterthemes


# In[72]:


dc.head()


# In[74]:


sns.countplot(dc['Incomeperyear'],palette='coolwarm',hue='sex',data=dc)


# In[75]:


dc["occupation"].value_counts().plot.pie(y=dc["occupation"],figsize=(15,15),autopct='%1.1f%%')


# In[76]:


dc["education"].value_counts().plot.pie(y=dc["education"],figsize=(15,15),autopct='%1.1f%%')


# In[82]:


plt.figure(figsize=(20,20))
sns.countplot(dc['marital_status'],palette='coolwarm',hue='sex',data=dc)


# In[79]:


sns.countplot(dc['Incomeperyear'],palette='coolwarm',hue='race',data=dc)


# In[ ]:




