import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("2020.csv",encoding="cp1252")
gk=['potential','age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gkf = pd.DataFrame(df, columns = gk)
golk=['potential','age','overall','position','gkdiving','gkhandling','gkkicking','gkpositioning','gkreflexes']
gk2=gkf = pd.DataFrame(gkf, columns = golk)
ft=gk2.loc[gk2['position']=='GK']
dt=ft.drop('position',axis=1)
y=dt['potential']
X=dt.drop('potential',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 40)
lr = LinearRegression()
lr.fit(X_train, y_train) 
pred_train_lr= lr.predict(X_train)
#print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
#print(r2_score(y_train, pred_train_lr))
pred_test_lr= lr.predict(X_test)
#print(np.sqrt(mean_squared_error(y_test,pred_test_lr))) 
#print(r2_score(y_test, pred_test_lr))

