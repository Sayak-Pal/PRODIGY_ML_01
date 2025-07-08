
"""# Import the library"""

import pandas as pa
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.rcParams["figure.figsize"] = (20, 10)

from sklearn.model_selection import train_test_split

"""# Read Data set"""

import os

df = pa.read_csv('train.csv')
display(df.head())

"""# Statistical of dataset"""

df.describe()

df.info()

"""# Check the missing value in percentage"""

missing_percentage = (df.isnull().sum() / len(df)) * 100
filtered_missing_percentage = missing_percentage[missing_percentage > 0]
print(filtered_missing_percentage)

"""1. Dropping values who have more than 30  """

df2 = df.drop('Id', axis=1)
df2.head(3)

df2=df.drop(['Alley', 'MasVnrType', 'FireplaceQu', 'PoolQC','Fence','MiscFeature', ], axis='columns')
df2.head()

"""2. Drop null value"""

df2 = df2.dropna()

"""3. Graphical representation to check null value"""

sb.heatmap(df2.isnull())

df.shape

df3=df2
df3.isnull().sum()

df3['LotArea'].unique()

df3['LotArea'] = df3['LotArea'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else x)
print(df3.head())

df3['LotArea'].unique()

df3[df3['LotArea'] >20]

sb.histplot(df3['LotArea'] ,kde=True)

column_names = df3.columns
print(column_names)

"""## MSSubClass: The building class"""

df3['MSSubClass']

sb.histplot(df3['MSSubClass'] ,kde=True)

"""# Label encoding"""

from sklearn.preprocessing import LabelEncoder,StandardScaler

# label encding
for col in df3.columns:
    if df3[col].dtype == 'object' or df3[col].dtype=='category':
        df3[col]=LabelEncoder().fit_transform(df3[col])
df3.head()

"""# Make the model"""



df3.head(10)

df3.columns

column_types = df3.dtypes
print(column_types)

X = df3.drop(['SalePrice'], axis=1)
X.head(3)

y=df3['SalePrice']

df.isnull().sum()[df.isnull().sum() > 0].sort_values(ascending=False)
missing_data_cols = df.isnull().sum()[df.isnull().sum() > 0].index.tolist()
missing_data_cols

# find only categorical columns
cat_cols = df3.select_dtypes(include='object').columns.tolist()
# find only numerical columns
num_cols = df3.select_dtypes(exclude='object').columns.tolist()

print(f'Categorical Columns: {cat_cols}')
print(f'Numerical Columns: {num_cols}')



from sklearn.linear_model import LinearRegression

X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model=LinearRegression()
model.fit(X_train, y_train)
model.score(X_test,y_test)

"""# Result"""

from sklearn.metrics import mean_squared_error, r2_score

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")