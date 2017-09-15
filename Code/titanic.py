#from sklearn.cross_validation import cross_val_predict
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
df = pd.read_csv('S:\\Kaggle\\Titanic\\train.csv',header=0)
cols = ['Name','Ticket','Cabin']
df = df.drop(cols,axis=1)
df = df.dropna()
#df.info()
dummies = []
cols = ['Pclass','Sex','Embarked']
for col in cols:
 dummies.append(pd.get_dummies(df[col]))
titanic_dummies = pd.concat(dummies, axis=1)
#df.info()
df = pd.concat((df,titanic_dummies),axis=1)
df = df.drop(['Pclass','Sex','Embarked'],axis=1)
#df.info()
df['Age'] = df['Age'].interpolate()
X = df.values
df.info()

#y_results = clf.predict(X)
y = df['Survived'].values
X = np.delete(X,1,axis=1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
from sklearn import tree
clf = tree.DecisionTreeClassifier(max_depth=5)
print(clf.fit(X_train,y_train))
print(clf.score(X_test,y_test))
print(clf.score(X_train,y_train))
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit (X_train, y_train)
print(clf.score (X_test, y_test))

clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))

clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))


#output = np.column_stack((X_results[:,0],y_results))
#df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
#df_results.to_csv('titanic_results.csv',index=False)
