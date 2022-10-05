import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg' , 'plas' , 'pres', 'skin' , 'test', 'mass' , 'pedi', 'age', 'class']

df  = pd.read_csv(url , names=names)

print(df)

# array = df.values
X = df.drop('class',axis=1)
y = df['class']

X_train,X_test, y_train, y_test = model_selection.train_test_split(X,y , test_size = 0.2, random_state=101)

# # fit the model
model = LogisticRegression()
model.fit(X_train , y_train)

# #accuracy

result = model.score(X_test , y_test)
print(f'the accuracy of the model is {result}')

import joblib
joblib.dump(model,'diabetes.pkl')