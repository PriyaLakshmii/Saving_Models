import joblib
model=joblib.load(open('diabetes.pkl','rb'))
result=model.predict([[1,1,1,1,1,1,1,1,]])[0]
print(result)