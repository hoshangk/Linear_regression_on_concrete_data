import pandas as pd
data = pd.read_csv('concrete.csv')    #Reading the Data
data.head()							  #See How the Data Look Like
data.count()							
data.describe()							#Will give all the details/statistics of the data



#Mean of the Stregth of the Cement is 35.81
#============================Preprocessing the Data==================	
data.isna().sum()					#Checking whether the data have no values

#============================Scaling the Numerical Values======================
from sklearn.preprocessing import RobustScaler
rs = RobustScaler()				#importing the RobustScaler

#numerical_feature = data[['cement','fineagg','coarseagg']]
data['cement'] = rs.fit_transform(data[['cement']])
data.head()
data['fineagg'] = rs.fit_transform(data[['fineagg']])
data['coarseagg'] = rs.fit_transform(data[['coarseagg']])
data['age'] = rs.fit_transform(data[['age']])

#================================Splitting the data set into train and test data===========
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


#==========================Use LinearRegression Model for prediction====================
from sklearn.linear_model import LinearRegression

LR = LinearRegression()     #creating the object of the model

LR.fit(X_train, y_train)	#feeding the train data to the model to train the model	
	
#==========================Output=========================================
#LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

y_predict = LR.predict(X_test)    			#Predincting the output for the train data

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict})   # Combining test_output & predicted output

df.head(10)

# Evaluating the Algorithm
import numpy as np
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))



Mean Absolute Error: 8.076637164748812
Mean Squared Error: 105.36431354685507
Root Mean Squared Error: 10.26471205377214


#RMS value from the above model is 10.264 which is >= 10% of mean value og the strength which is
#35.817. Which shows that Linear Regression model is not suitable to predict the strength of the 
#cement

