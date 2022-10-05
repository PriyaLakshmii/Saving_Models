import joblib
model=joblib.load('diabetes.pkl')
result=model.predict([[1,1,1,1,1,1,1,1,]])[0]
print(result)