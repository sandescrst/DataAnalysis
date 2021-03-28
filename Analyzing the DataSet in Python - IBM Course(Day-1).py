#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing a CSV into Python
import pandas as pd
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
df = pd.read_csv(url, header = None)
df.head()


# In[2]:


# create headers list
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]


# In[3]:


print("headers\n", headers)


# In[4]:


df.columns = headers


# In[5]:


df.head(10)


# In[6]:


df.dropna(subset=["price"], axis=0)


# In[7]:


print(df.columns)


# In[8]:


path = "C:/Users/sande/Documents/Analyzing Datasets- IBM Course/automobile.csv"
df.to_csv(path, index = False)


# In[9]:


df.dtypes


# In[10]:


df.describe()


# In[11]:


# describe all the columns in "df" 
df.describe(include = "all")


# In[12]:


# Write your code below and press Shift+Enter to execute 
df[["length", "compression-ratio"]].describe()


# In[13]:


# look at the info of "df"
df.info


# In[14]:


import matplotlib.pylab as plt


# In[15]:


import numpy as np


# In[16]:


#replace the "?" with "NaN"
df.replace("?", np.nan, inplace =True)
df.head(5)


# In[17]:


#to find the missing value in the dataset we use either "isnull()" or ".notnull()".
#output will be in boolean
missing_data = df.isnull()
missing_data.head(5)


# In[20]:


#for loop helps us to quick count the missing values in the each column. 
#where "True" denotes there is missing value where as "False" denotes no missing values 
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")


# In[21]:


# Based on the summary above, each column has 205 rows of data, 
#seven columns containing missing data:

# "normalized-losses": 41 missing data
# "num-of-doors": 2 missing data
# "bore": 4 missing data
# "stroke" : 4 missing data
# "horsepower": 2 missing data
# "peak-rpm": 2 missing data
# "price": 4 missing data


# In[22]:


#DEAING WITH MISSING VALUES
# Replace by mean:

# "normalized-losses": 41 missing data, replace them with mean
# "stroke": 4 missing data, replace them with mean
# "bore": 4 missing data, replace them with mean
# "horsepower": 2 missing data, replace them with mean
# "peak-rpm": 2 missing data, replace them with mean

# Replace by frequency:

# "num-of-doors": 2 missing data, replace them with "four".
# Reason: 84% sedans is four doors. Since four doors is most frequent, it is most likely to occur

# Drop the whole row:

# "price": 4 missing data, simply delete the whole row
# Reason: price is what we want to predict. 
#Any data entry without price data cannot be used for prediction; 
#therefore any row now without price data is not useful to us


# In[23]:


#calulating the mean of the column normalized losses
avg_norm = df["normalized-losses"].astype("float").mean(axis=0)
print("avg of normalized losses is:", avg_norm)


# In[24]:


#replacing the missing values with average 
df["normalized-losses"].replace(np.nan, avg_norm, inplace = True) 


# In[25]:


avg_stroke = df["stroke"].astype("float").mean(axis=0)
print("avg of stroke is:", avg_stroke)


# In[26]:


df["stroke"].replace(np.nan, avg_stroke, inplace = True) 


# In[27]:


avg_bore= df["bore"].astype("float").mean(axis=0)
print("avg of stroke is:", avg_bore)


# In[28]:


df["bore"].replace(np.nan, avg_bore, inplace = True) 


# In[29]:


avg_hp = df["horsepower"].astype("float").mean(axis=0)
print("avg of horsepower is:", avg_hp)


# In[30]:


df["horsepower"].replace(np.nan, avg_hp, inplace = True) 


# In[31]:


avg_rpm = df["peak-rpm"].astype("float").mean(axis=0)
print("avg of rpm is:", avg_rpm)


# In[32]:


df["peak-rpm"].replace(np.nan, avg_rpm, inplace = True) 


# In[34]:


#To see which values are present in a particular column,
#we can use the ".value_counts()" method:

df["num-of-doors"].value_counts()


# In[35]:


#We can see that four doors are the most common type. 
#We can also use the ".idxmax()" method to calculate
#for us the most common type automatically:
df["num-of-doors"].value_counts().idxmax()


# In[36]:


#replacing the missing value with the frequecy
df["num-of-doors"].replace(np.nan, "four", inplace = True) 


# In[38]:


#dropping the missing values for the price
df.dropna(subset =["price"], axis=0, inplace= True)
# we need to reset index because we drop the four rows
df.reset_index(drop =True, inplace =True)


# In[40]:


df.head(20)


# In[41]:


df.dtypes


# In[46]:


#converting the data types according to thier values
df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")


# In[49]:


df[["normalized-losses"]] = df[["normalized-losses"]].astype("int64")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")


# In[50]:


df.dtypes


# In[51]:


df.head()


# In[52]:


# Convert mpg to L/100km by mathematical operation (235 divided by mpg)
df["city-L/100km"] = 235/df["city-mpg"]


# In[53]:


df.head()


# In[58]:


#Binning in pandas
df["horsepower"] = df["horsepower"].astype(int, copy=True)


# In[59]:


#matplotlib library
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("Horsepower Bins")


# In[63]:


bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]),4)
bins
group_names = ["Low","Medium","High"]


# In[ ]:





# In[66]:


#binned horsepower column and it gets added into the df as well
df["horsepower-binned"] = pd.cut(df["horsepower"], bins, labels=group_names, include_lowest =True)
df[["horsepower","horsepower-binned"]].head(20)


# In[67]:


df["horsepower-binned"].value_counts()


# In[75]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.bar(group_names, df["horsepower-binned"].value_counts())

#giving x/y labels to the plot
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower Bins")


# In[76]:


#Bins visualization
#Normally, a histogram is used to visualize
#the distribution of bins we created above.


# In[84]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as plt
from matplotlib import pyplot

# draw historgram of attribute "horsepower" with bins = 3
plt.pyplot.hist(df["horsepower"], bins = 3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("Horsepower visualization")


# In[86]:


df.columns


# In[90]:


dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.tail(5)


# In[94]:


dummy_variable_1.rename(columns = {"fuel-type-gas":"gas", "fuel-type-diesel": "diesel"}, inplace = True)


# In[95]:


#in order to add the column dummy_variable_1 in the dataset we need to concat
df = pd.concat([df, dummy_variable_1], axis=1)
df.drop("fuel-type", axis = 1, inplace= True)


# In[96]:


df.head()


# In[97]:


# replace (original value) by (original value)/(maximum value)
df['length'] = df['length']/df['length'].max()
df['width'] = df['width']/df['width'].max()


# In[98]:


df['height'] = df['height']/df['height'].max()


# In[99]:


df.head()


# In[100]:


dummy_var_2 = pd.get_dummies(df["aspiration"], )


# In[101]:


df.head()


# In[103]:


dummy_var_2.head()


# In[104]:


dummy_var_2.rename(columns = {"aspiration-std":"std", "aspiration-turbo":"turbo"}, inplace = True)


# In[105]:


dummy_var_2.head()


# In[106]:


df = pd.concat([df,dummy_var_2], axis=1)


# In[107]:


df.drop("aspiration", axis=1, inplace = True)


# In[108]:


df.head(5)


# In[111]:


#this save the new file 
df.to_csv("C:/Users/sande/Documents/Analyzing Datasets- IBM Course/automobile_Cleanofftest.csv")


# In[ ]:




