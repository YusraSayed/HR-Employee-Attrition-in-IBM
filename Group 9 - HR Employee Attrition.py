#!/usr/bin/env python
# coding: utf-8

# ## Business Case:-Based on given features we need to find whether an employee will leave the company or not.

# !pip install hvplot

# In[1]:


## Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


## Loading the data
data=pd.read_csv('HR-Employee-Attrition.csv')
data.head()


# # EDA

# In[58]:


data.columns


# In[63]:


data.tail()


# # Univariant Analysis

# In[4]:


import sweetviz as sv #importing sweetviz library 
my_report = sv.analyze(data) #syntax to use sweetviz
my_report.show_html()
#Default arguments will generate to "SWEETVIZ_REPORT.html"


# In[3]:


## Create a new dataframe with categorical variables only(Check the datatype by using info function)
data1=data[['BusinessTravel',
 'Department',
 'EducationField',
 'Gender',
 'JobRole',
 'MaritalStatus',
 'Over18',
 'OverTime']]


# In[64]:


# Plotting how every  categorical feature correlate with the "target"
plt.figure(figsize=(50,50), facecolor='white')#canvas size
plotnumber = 1          #count variable

for column in data1:          #for loop to acess columns form data1
    if plotnumber<=16 :       #checking whether count variable is less than 16 or not
        ax = plt.subplot(4,4,plotnumber)#plotting 8 graphs in canvas(4 rows and 4 columns)
        sns.countplot(x=data1[column].dropna(axis=0)#plotting count plot 
                        ,hue=data.Attrition)
        plt.xlabel(column,fontsize=25)#assigning name to x-axis and increasing it's font 
        plt.ylabel('Attrition',fontsize=20)#assigning name to y-axis and increasing it's font 
    plotnumber+=1#increasing counter
plt.tight_layout()


# In[4]:


continous_col = []#list for continous columns
for column in data.columns:#acessing columns from datasets
    if data[column].dtype == int and len(data[column].unique()) >= 10: #checking whether it's datatype is int and count of unique label greater than 10  
        continous_col.append(column) # inserting those columns in list                                      
continous_col


# In[5]:


## discrete data
discrete_col = []#list for continous columns
for column in data.columns:#acessing columns from datasets
    if data[column].dtype == int and len(data[column].unique()) < 10: #checking whether it's datatype is int and count of unique label greater than 10  
        discrete_col.append(column)
discrete_col


# In[6]:


# columns with discrete values
data3=data[['Education',
 'EmployeeCount',
 'EnvironmentSatisfaction',
 'JobInvolvement',
 'JobLevel',
 'JobSatisfaction',
 'NumCompaniesWorked',
 'PerformanceRating',
 'RelationshipSatisfaction',
 'StandardHours',
 'StockOptionLevel',
 'TrainingTimesLastYear',
 'WorkLifeBalance']]


# In[10]:


# Plotting how every  discrete feature correlate with the "target"
plt.figure(figsize=(20,25), facecolor='white')#canvas size
plotnumber = 1

for column in data3:
    if plotnumber<=16 :
        ax = plt.subplot(4,4,plotnumber)
        sns.countplot(x=data3[column].dropna(axis=0) ,hue=data.Attrition)
        plt.xlabel(column,fontsize=20)
        plt.ylabel('Attrition',fontsize=20)
    plotnumber+=1
plt.tight_layout()


# ### Bivariant analysis of continuous variables

# In[7]:


# columns with continuous variables/columns
data2=data[['Age',
 'DailyRate',
 'DistanceFromHome',
 'EmployeeNumber',
 'HourlyRate',
 'MonthlyIncome',
 'MonthlyRate',
 'NumCompaniesWorked',
 'PercentSalaryHike',
 'TotalWorkingYears',
 'YearsAtCompany',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager']]


# In[12]:


# Plotting how every  numerical feature correlate with the "target"
plt.figure(figsize=(20,25), facecolor='white')#canvas size
plotnumber = 1#counter for number of plot

for column in data2:#acessing columns form data2 DataFrame
    if plotnumber<=16 :#checking whether counter is less than 16 or not
        ax = plt.subplot(4,4,plotnumber)#plotting 8 graphs in canvas(4 rows and 4 columns)
        sns.histplot(x=data2[column].dropna(axis=0)# plotting hist plot and dropping null values,classification according to target
                        ,hue=data.Attrition)
        plt.xlabel(column,fontsize=20)##assigning name to x-axis and increasing it's font 
        plt.ylabel('Attrition',fontsize=20)#assigning name to y-axis and increasing it's font 
    plotnumber+=1#increasing counter by 1
plt.tight_layout()


# In[14]:


sns.barplot(x='Attrition', y='DistanceFromHome', data=data)


# In[15]:


sns.barplot(x='Attrition', y='Age', data=data)


# In[17]:


sns.barplot(x='Attrition', y='NumCompaniesWorked', data=data)


# In[30]:


sns.boxplot(x="Attrition",y="Age",data=data,palette=["#D4A1E7","#6faea4"])


# In[35]:


pie = data.groupby('Attrition')['Attrition'].count()
plt.pie(pie, explode=[0.1, 0.1], labels=['No', 'Yes'], autopct='%1.1f%%')
plt.title("% of Attrition")
plt.show()
# 84% of the employees in the dataset have not left the company.


# In[41]:


plt.figure(figsize=(8,5))
sns.barplot(x='JobRole', y='MonthlyIncome', hue='Attrition', data=data)
plt.xticks(rotation=90)
plt.show()


# ## Insights
# 
# BusinessTravel : The workers who travel alot are more likely to quit then other employees.
# 
# Department : The worker in Research & Development are more likely to stay then the workers on other departement.
# 
# EducationField : The workers with Human Resources and Technical Degree are more likely to quit then employees from other fields of educations.
# 
# Gender : The Male are more likely to quit.
# 
# JobRole : The workers in Laboratory Technician, Sales Representative, and Human Resources are more likely to quit the workers in other positions.
# 
# MaritalStatus : The workers who have Single marital status are more likely to quit the Married, and Divorced.
# 
# OverTime : Attrition rate is almost equal

# In[8]:


data.isnull().sum()#null value checking 
# no null values


# In[9]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=False)


# # Data Preprocessing

# ###  1.Attrition

# In[10]:


# Conversion of  Categorical variables
## Manual encoding Attrition feature
data.Attrition=data.Attrition.map({'Yes':1,'No':0})
data.head()###  1.Attrition


# ###  2.BusinessTravel 

# In[11]:


data.BusinessTravel=data.BusinessTravel.map({'Travel_Frequently':2,
                                             'Travel_Rarely':1,
                                             'Non-Travel':0})


# ### 3.Department

# In[12]:


data.Department=data.Department.map({'Research & Development':2,
                                     'Sales':1,
                                     'Human Resources':0})#imputation using map function
data.head()


# ### 4.EducationField

# In[13]:


#using map function
data.EducationField=data.EducationField.map({'Life Sciences':5,
                                             'Medical':4,
                                             'Marketing':3,
                                             'Technical Degree':2,
                                             'Other':1,
                                             'Human Resources':0 })


# ### 5.Gender

# In[14]:


#checking weightage of each label whoever have high count 
data.Gender.value_counts()


# In[15]:


## Encoding Gender by one hot encoding.
data.Gender=pd.get_dummies(data.Gender,drop_first=True)


# In[16]:


data.Gender


# ### 6. JobRole

# In[17]:


## Encoding JobRole
data.JobRole=data.JobRole.map({'Laboratory Technician':8,
                               'Sales Executive':7,
                               'Research Scientist':6,
                               'Sales Representative':5,
                               'Human Resources':4,
                               'Manufacturing Director':3,
                               'Healthcare Representative':2,
                               'Manager':1,
                               'Research Director':0 })


# ### 7. Encoding MaritalStatus using label encoding 
# 

# In[18]:


## Encoding MaritalStatus

from sklearn.preprocessing import LabelEncoder
#importing label encoder from sklearn 

label = LabelEncoder()#object creation 

data.MaritalStatus=label.fit_transform(data.MaritalStatus)
#applying label encoder to  marital status

data.MaritalStatus


# ### 8. OverTime

# In[19]:


## Encoding OverTime
data.OverTime=label.fit_transform(data.OverTime)#label encoding

data.OverTime


# ### `Employee attrition is a major concern for companies, but by using data analysis, companies can gain valuable insights into why employees leave and take proactive measures to improve retention. The IBM HR Dataset is a valuable resource for this, providing a wide range of variables that can help companies understand the root causes of attrition and improve employee engagement. By using this dataset, companies can reduce turnover rates, improve productivity, and create a positive work environment for their employees`

# ## Implementation of ML Algorithm

# In[37]:


# Dropping 'Over 18' column because values are constant
data.drop(['Over18'], inplace = True, axis = 1)


# In[39]:


## Creating independent and dependent variable
X = data.drop('Attrition', axis=1)#independent variable 
y = data.Attrition#dependent variable 


# In[40]:


## preparing training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.25, random_state=42)


# In[41]:


#importing decision tree from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

#object creation for decision tree
dt=DecisionTreeClassifier()

#training the model  
dt.fit(X_train, y_train)

#prediction
y_predict=dt.predict(X_test)


# In[42]:


#predicting training data to check training performance.
y_train_predict=dt.predict(X_train)
y_train_predict


# ## Evaluation

# In[52]:


#importing mertics to check model performance
from sklearn.metrics import accuracy_score,f1_score, classification_report

##Training score
y_train_predict = dt.predict(X_test)

acc=accuracy_score(y_test,y_predict)
acc


# In[53]:


f1=f1_score(y_test,y_predict)
f1


# In[54]:


cm1=pd.crosstab(y_test,y_predict)
cm1


# In[55]:


print(classification_report(y_test,y_predict))

