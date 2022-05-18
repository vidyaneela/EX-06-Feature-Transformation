## EX-06-Feature-Transformation
# AIM
To Perform the various feature transformation techniques on a dataset and save the data to a file.

# Explanation
Feature Transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
STEP 1
Read the given Data

STEP 2
Clean the Data Set using Data Cleaning Process

STEP 3
Apply Feature Transformation techniques to all the feature of the data set

STEP 4
Save the data to the file

# CODE:
Developed by:212221230120
# Data_to_Transform.csv :
```
# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal')

df=pd.read_csv("Data_to_Transform.csv")
df

#checking and analysing the data
df.isnull().sum()

#checking for skewness of data
df.skew()

#applying data transformations
dfmp=pd.DataFrame()
#for Moderate Positive Skew
#function transformation
dfmp["Moderate Positive Skew"]=df["Moderate Positive Skew"]
dfmp["MPS_log"]=np.log(df["Moderate Positive Skew"]) 
dfmp["MPS_rp"]=np.reciprocal(df["Moderate Positive Skew"])
dfmp["MPS_sqr"]=np.sqrt(df["Moderate Positive Skew"])
#power transformation
dfmp["MPS_yj"], parameters=stats.yeojohnson(df["Moderate Positive Skew"])
dfmp["MPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
#quantile transformation
dfmp["MPS_qt"]=qt.fit_transform(df[["Moderate Positive Skew"]])
dfmp.skew()

dfmp.drop('MPS_rp',axis=1,inplace=True)
dfmp.skew()

dfmp

#for Highly Positive Skew
#function transformation
dfhp=pd.DataFrame()
dfhp["Highly Positive Skew"]=df["Highly Positive Skew"]
dfhp["HPS_log"]=np.log(df["Highly Positive Skew"]) 
dfhp["HPS_rp"]=np.reciprocal(df["Highly Positive Skew"])
dfhp["HPS_sqr"]=np.sqrt(df["Highly Positive Skew"])
#power transformation
dfhp["HPS_yj"], parameters=stats.yeojohnson(df["Highly Positive Skew"])
dfhp["HPS_bc"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
#quantile transformation
dfhp["HPS_qt"]=qt.fit_transform(df[["Highly Positive Skew"]])
dfhp.skew()

dfhp.drop('HPS_sqr',axis=1,inplace=True)
dfhp.skew()

dfhp

#for Moderate Negative Skew
dfmn=pd.DataFrame()
#function transformation
dfmn["Moderate Negative Skew"]=df["Moderate Negative Skew"]
dfmn["MNS_rp"]=np.reciprocal(df["Moderate Negative Skew"])
dfmn["MNS_sq"]=np.square(df["Moderate Negative Skew"])
#power transformation
dfmn["MNS_yj"], parameters=stats.yeojohnson(df["Moderate Negative Skew"]) 
#quantile transformation
dfmn["MNS_qt"]=qt.fit_transform(df[["Moderate Negative Skew"]])
dfmn.skew()

dfmn.drop('MNS_rp',axis=1,inplace=True)
dfmn.skew()

dfmn

#for Highly Negative Skew
dfhn=pd.DataFrame()
#function transformation
dfhn["Highly Negative Skew"]=df["Highly Negative Skew"]
dfhn["HNS_rp"]=np.reciprocal(df["Highly Negative Skew"])
dfhn["HNS_sq"]=np.square(df["Highly Negative Skew"])
#phwer transformation
dfhn["HNS_yj"], parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
#quantile transformation
dfhn["HNS_qt"]=qt.fit_transform(df[["Highly Negative Skew"]])
dfhn.skew()

dfhn.drop('HNS_rp',axis=1,inplace=True)
dfhn.skew()

dfhn

#graphical representation
#for Moderate Positive Skew
df["Moderate Positive Skew"].hist()
dfmp["MPS_log"].hist()
dfmp["MPS_sqr"].hist()
dfmp["MPS_bc"].hist()
dfmp["MPS_yj"].hist()
sm.qqplot(df['Moderate Positive Skew'],line='45')
plt.show()
sm.qqplot(dfmp['MPS_qt'],line='45')
plt.show()

#for Highly Positive Skew
df["Highly Positive Skew"].hist()
dfhp["HPS_log"].hist()
dfhp["HPS_rp"].hist()
dfhp["HPS_bc"].hist()
dfhp["HPS_yj"].hist()
sm.qqplot(df['Highly Positive Skew'],line='45')
plt.show()
sm.qqplot(dfhp['HPS_qt'],line='45')
plt.show()

#for Moderate Negative Skew
df["Moderate Negative Skew"].hist()
dfmn["MNS_sq"].hist()
dfmn["MNS_yj"].hist()
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
sm.qqplot(dfmn['MNS_qt'],line='45')
plt.show()

#for Highly Negative Skew
df["Highly Negative Skew"].hist()
dfhn["HNS_sq"].hist()
dfhn["HNS_yj"].hist()
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
sm.qqplot(dfhn['HNS_qt'],line='45')
plt.show()
```
# OUPUT:
![o1](https://user-images.githubusercontent.com/94169318/169075407-45a367df-f127-4aab-abff-4b4b3ff90691.png)
![02](https://user-images.githubusercontent.com/94169318/169075508-860dd015-637c-4f7c-a3ac-bf22fa1dc193.png)
![o3](https://user-images.githubusercontent.com/94169318/169076099-77354301-6a7c-413e-9549-39b21844dc04.png)
![o4](https://user-images.githubusercontent.com/94169318/169076161-79a7fef1-f999-4919-bbc9-d47550bd93ce.png)
![o5](https://user-images.githubusercontent.com/94169318/169076198-d6530d7f-5949-4224-a98f-4e514e3f460c.png)
![o6](https://user-images.githubusercontent.com/94169318/169076260-4e526a0a-c492-4ce1-a4f1-bf70919c4390.png)
![o7](https://user-images.githubusercontent.com/94169318/169076304-53aa700f-e32f-44ad-85c3-1a6032834441.png)
![o8](https://user-images.githubusercontent.com/94169318/169076351-c20337d9-9b36-4d69-87b3-939fe0790867.png)
![o9](https://user-images.githubusercontent.com/94169318/169076392-2db5f2d2-7e36-4ff8-a00a-6689db198252.png)
![o10](https://user-images.githubusercontent.com/94169318/169076444-480db067-e170-4a39-8518-9e0d4a54a8bb.png)
![o11](https://user-images.githubusercontent.com/94169318/169076601-c0a50055-cafa-4744-85e5-51968e99efc4.png)
![o12](https://user-images.githubusercontent.com/94169318/169076648-37662a53-41f7-4a2d-a742-806142465474.png)
![o13](https://user-images.githubusercontent.com/94169318/169076700-f549c039-e54e-407a-9d8e-fc86e77d089d.png)
![o14](https://user-images.githubusercontent.com/94169318/169076766-9f8e67bb-f5a0-4d0d-9b50-4fea5234f095.png)
![o15](https://user-images.githubusercontent.com/94169318/169076799-739fe284-3be2-47ea-9f11-c56447a97f37.png)


# For Titanic_dataset.csv:
```
# importing packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df=pd.read_csv("titanic_dataset.csv")
df

#checking and analysing the data
df.isnull().sum()
# cleaning data
df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()
#encoding categorical data
from sklearn.preprocessing import OrdinalEncoder
embark=["C","S","Q"]
emb=OrdinalEncoder(categories=[embark])
df["Embarked"]=emb.fit_transform(df[["Embarked"]])

from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
df.skew()

#feature in 0.5 to -0.5 range: Embarked Survived Age (A,S pos)(E neg)
#features that are in skew
#neg:Pclass 
#pos:Sex SibSp Parch Fare 
df["Age_1"]=qt.fit_transform(df[["Age"]])
df["Survived_1"]=qt.fit_transform(df[["Survived"]])
df["Embarked_1"]=qt.fit_transform(df[["Embarked"]])
df["Pclass_sq"]=np.square(df["Pclass"])
df["Pclass_qt"]=qt.fit_transform(df[["Pclass"]])
df["SibSp_yj"], parameters=stats.yeojohnson(df["SibSp"])
df["SibSp_qt"]=qt.fit_transform(df[["SibSp"]])

df["Parch_yj"], parameters=stats.yeojohnson(df["Parch"])
df["Parch_qt"]=qt.fit_transform(df[["Parch"]])

df["Fare_yj"], parameters=stats.yeojohnson(df["Fare"])
df["Fare_qt"]=qt.fit_transform(df[["Fare"]])

df["Sex_yj"], parameters=stats.yeojohnson(df["Sex"])
df["Sex_qt"]=qt.fit_transform(df[["Sex"]])
df.skew()

#taking closer to range skew values
df.drop('Sex_yj',axis=1,inplace=True)
df.drop('Pclass_qt',axis=1,inplace=True)
df.drop('SibSp_qt',axis=1,inplace=True)
df.drop('Parch_qt',axis=1,inplace=True)
df.drop('Fare_qt',axis=1,inplace=True)
df.skew()

#graph representation
df["Sex"].hist()
df["Sex_qt"].hist()
df["SibSp"].hist()
df["SibSp_yj"].hist()
df["Parch"].hist()
df["Parch_yj"].hist()
df["Fare"].hist()
df["Fare_yj"].hist()
df["Pclass"].hist()
df["Pclass_sq"].hist()
```
# Output:
![o16](https://user-images.githubusercontent.com/94169318/169077212-0a00798a-e1cf-4d55-928e-593d561577ab.png)
![o20](https://user-images.githubusercontent.com/94169318/169077264-15fceee7-9091-45c3-9dec-3bc64c3ad590.png)
![o18](https://user-images.githubusercontent.com/94169318/169077385-a7bce00f-0ec8-46b1-91f7-e991c91a9ecf.png)
![o21](https://user-images.githubusercontent.com/94169318/169077431-82e1993d-b097-48e0-8e80-e82e8ea40541.png)
![o22](https://user-images.githubusercontent.com/94169318/169077466-97afb0d7-f06c-40f4-b2c1-d48822f9aea8.png)
![o23](https://user-images.githubusercontent.com/94169318/169077529-1ce488de-2ed9-42b5-9293-663a37c9cf0d.png)
![o24](https://user-images.githubusercontent.com/94169318/169077584-24652814-1c1d-4e8c-a565-65b4b5fbadf2.png)
![o25](https://user-images.githubusercontent.com/94169318/169077642-11e43a25-2551-4cf0-bbd9-ec1b2a360f1f.png)

# RESULT:
The various feature transformation techniques has been performed on the given datasets and the data are saved to a file
