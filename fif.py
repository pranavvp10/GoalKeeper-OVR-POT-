# importing important libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df=pd.read_csv("2020.csv",encoding="cp1252") #loading dataset

gk=['potential','age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gkf = pd.DataFrame(df, columns = gk)
golk=['age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gk2=gkf = pd.DataFrame(gkf, columns = golk)
ft=gk2.loc[gk2['position']=='GK']
dt=ft.drop('position',axis=1)

y=dt['overall'] #target variable
X=dt.drop('overall',axis=1) #predictor variables

#splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)

#fitting or training the model
lr = LinearRegression()
lr.fit(X_train, y_train)

pred_train_lr= lr.predict(X_train)
pred_test_lr= lr.predict(X_test)



