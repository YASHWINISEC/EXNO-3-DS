EXNO-3-DS
AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

FEATURE ENCODING:
Ordinal Encoding An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
Label Encoding Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
Binary Encoding Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
One Hot Encoding We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.
Methods Used for Data Transformation:
1. FUNCTION TRANSFORMATION
• Log Transformation • Reciprocal Transformation • Square Root Transformation • Square Transformation

2. POWER TRANSFORMATION
• Boxcox method • Yeojohnson method

CODING AND OUTPUT:
/* DEVELOPED BY :MONISH S
REGISTER NO: 212223040115*/
import pandas as pd
df = pd.read_csv("Encoding_data.csv")
df.head()
image

df.tail()
image

df.describe()
image

df.info()
image

df.shape
image

df
image

#ordinal encoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot', 'Warm','Cold']
oe=OrdinalEncoder(categories=[pm])
oe.fit_transform(df[["ord_2"]])
image

df['bo2']=oe.fit_transform(df[["ord_2"]])
df
image

#label Encoder
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
image

#One hot encoder
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse = False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
image

pd.get_dummies(df2,columns=["nom_0"])
image

pip install --upgrade category_encoders
from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
image

#binary encoder
be = BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb

image

#target encoder
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new = te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
image

#Feature Transformation
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("Data_to_Transform.csv")
df
image

df.info()
image

df.describe()
image

df.size
image

df.skew()
image

np.log(df["Highly Positive Skew"])
image

np.reciprocal(df["Moderate Positive Skew"])
image

np.sqrt(df["Highly Positive Skew"])
image

np.square(df["Highly Positive Skew"])
image

df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
image

df["Moderate Negative Skew_yeojohnson"],parameters =stats.yeojohnson(df["Moderate Negative Skew"])
df
image

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import scipy.stats as stats
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
image

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])


sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
image

df
image

df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df['Moderate Negative Skew'],line='45')
plt.show()
image

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
image

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
image

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
image

RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.
