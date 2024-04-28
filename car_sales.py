import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
from math import *
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


sales_df = pd.read_csv(r'C:\Users\USER\Downloads\Car Sales.xlsx - car_data.csv')
print(sales_df.info())
print(sales_df.head())
print(sales_df.isnull().sum())
sales_df=sales_df.dropna()
print(sales_df.isnull().sum())
print(sales_df.duplicated().sum())
sales_df=sales_df.drop_duplicates()
print(sales_df.duplicated().sum())
print(sales_df.head())
print(sales_df.nunique())

#FIND THE COUNT OF COMPANY
sns.countplot(y="Company",hue="Company", data = sales_df)
plt.title("frequency of each company")
plt.show()   #OR
company_count= pd.DataFrame(sales_df["Company"].value_counts())
print(company_count)
sns.barplot(data=company_count,y='Company',x='count',hue='Company')
plt.title("frequency of each company")
plt.show()

#FIND THE AVG PRICE OF CAR FOR EACH COMPANY
sns.barplot(y="Company",x="Price ($)",data=sales_df,hue="Company")
plt.title("average price of cars per company")
plt.show()  #OR
avg_price_company = sales_df.groupby('Company')[["Price ($)"]].mean()
print(avg_price_company)
sns.barplot(x="Price ($)",y="Company",hue="Company",data=avg_price_company)
plt.title("average price of cars per company")
plt.show()

#FIND THE DISRIBUTION OF COMAPNY AND PRICE USING BOXPLOT
sns.boxplot(x="Price ($)",y="Company",hue="Company",data=sales_df)
plt.title("Price Distibution By Company")
plt.show()
#Frequency of transmission types
sns.countplot(x="Transmission",hue="Transmission",data=sales_df)
plt.title("Frequency of transmission types")
plt.show()
#Price Distibution By Transmission
sns.boxplot(x="Price ($)",y="Transmission",hue="Transmission",data=sales_df)
plt.title("Price Distibution By Transmission")
plt.show()
#Frequency of Engine types
sns.countplot(x="Engine",hue="Engine",data=sales_df)
plt.title("Frequency of Engine types")
plt.show()
#Price Distibution By Engine
sns.boxplot(x="Price ($)",y="Engine",hue="Engine",data=sales_df)
plt.title("Price Distibution By Engine")
plt.show()
#frequency of each Color
sns.countplot(x="Color",hue="Color", data = sales_df)
plt.title("frequency of each Color")
plt.show()
#Price distribution based on color
sns.boxplot(x="Price ($)",y="Color",hue="Color",data=sales_df)
plt.title("Price distribution based on color")
plt.show()
#frequency of each Body style
sns.countplot(x="Body Style",hue="Body Style", data = sales_df)
plt.title("frequency of each Body Style")
plt.show()
#Price distribution based on body style
sns.boxplot(x="Price ($)",y="Body Style",hue="Body Style",data=sales_df)
plt.title("Price distribution based on Body Style")
plt.show()
#frequency of each Region
sns.countplot(x="Dealer_Region",hue="Dealer_Region", data = sales_df)
plt.title("frequency of each Dealer Region")
plt.show()
#Price distribution based on region
sns.boxplot(x="Price ($)",y="Dealer_Region",hue="Dealer_Region",data=sales_df)
plt.title("Price distribution based on Dealer Region")
plt.show()
#Annual income distribution
sns.histplot(x="Annual Income",kde=True,data=sales_df)
plt.title("Annual income distribution")
plt.show()
#Price distribution
sns.histplot(x="Price ($)",bins=15,kde=True,data=sales_df)
plt.title("Price distribution")
plt.grid()
plt.show()
#Avg annual income based on Gender
sns.barplot(x="Gender",y="Annual Income",hue="Gender",data=sales_df)
plt.title("Avg Annual income based on Gender")
plt.show()
#Annual income distribution based on Gender
sns.boxplot(x="Annual Income",y="Gender",hue="Gender",data=sales_df)
plt.title("Annual income distribution based on Gender")
plt.show()
#relationship between gender and body style
cross_tab=pd.crosstab(sales_df["Gender"],sales_df["Body Style"])
print(cross_tab)
cross_tab.plot(kind='bar')
plt.title("relationship between gender and body style")
plt.show()
#relationship between gender and Transmission
cross_tab=pd.crosstab(sales_df["Gender"],sales_df["Transmission"])
print(cross_tab)
cross_tab.plot(kind='bar')
plt.title("relationship between gender and Transmission")
plt.show()
#relationship between gender and color
cross_tab=pd.crosstab(sales_df["Gender"],sales_df["Color"])
print(cross_tab)
cross_tab.plot(kind='bar')
plt.title("relationship between gender and Color")
plt.show()

#DETERMINE THE PRICE OF A CAR

#Dropping columns not needed for regression
sales_df=sales_df.drop(['Car_id','Date','Customer Name','Gender','Dealer_Name','Annual Income','Dealer_No ','Phone'],axis=1)
print(sales_df)

#rearranging the data 
column=sales_df.pop('Price ($)')
sales_df['Price ($)']=column
print(sales_df)

#Encoding
label = LabelEncoder()
sales_df['Model']=label.fit_transform(sales_df['Model'])
sales_df['Company']=label.fit_transform(sales_df['Company'])
sales_df['Engine']=label.fit_transform(sales_df['Engine'])
sales_df['Transmission']=label.fit_transform(sales_df['Transmission'])
sales_df['Color']=label.fit_transform(sales_df['Color'])
sales_df['Body Style']=label.fit_transform(sales_df['Body Style'])
sales_df['Dealer_Region']=label.fit_transform(sales_df['Dealer_Region'])

print(sales_df)

#DEFINING DEPENDENT AND INDEPENDENT VALUES
x=sales_df.iloc[:,:-1]
y= sales_df.iloc[:,-1]
print(x.shape)
print(y.shape)

#DATA SPLITTING
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)
#IMPORTING PACKAGES
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import sklearn.tree as tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor  
from catboost import CatBoostRegressor 
from lightgbm import LGBMRegressor  
from sklearn.neural_network import MLPRegressor 
from sklearn.metrics import mean_absolute_error, r2_score


Models=[('Linear Regression', LinearRegression()),
('Ridge Regression',Ridge()),
('Lasso Regression',Lasso()),
('Decision Tree Regressor',DecisionTreeRegressor()),
('Random Forest Regressor',RandomForestRegressor()),
('Hist GradientBoost Regressor', HistGradientBoostingRegressor()),
('Gradient Boosting Regressor',GradientBoostingRegressor()),
('Support Vector Machine',SVR()),
('XGBoost',XGBRegressor()),
('Catboost',CatBoostRegressor(verbose=False)),
('lightgbm',LGBMRegressor()),
('Multi Layer Perceptron',MLPRegressor())
]

for model_name,model in Models:
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    print("model:  {}".format(model_name))
    print("mae: {}".format(mae))
    print("r2: {}".format(r2))
    plt.scatter(y_test,y_pred)
    plt.plot([min(y_test),max(y_test)],[min(y_test),max(y_test)],color="red")
    plt.xlabel("Actual labels")
    plt.ylabel("Predicted labels")
    plt.title("True vs Predicted Values using R2 square")
    plt.show()
