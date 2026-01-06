#Import Required Libraries


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



#Load Dataset


data = pd.read_csv("loan_prediction_with_applicant_no.csv")
print(data.head())



#Data Preparation


X = data.drop(["ApplicantNo", "Loan_Status"], axis=1)
y = data["Loan_Status"]




#Train–Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=49
)




#Train Logistic Regression Model


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)




#Model Evaluation


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




#Predict New Loan Application


# Format:
# [ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Married, Education]

new_applicant = [[6000, 1500, 170, 360, 1, 1, 1]]

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Approved")
else:
    print("Loan Rejected")




#Sample Output


#Accuracy: 0.85

#Confusion Matrix:
#[[5 1]
# [0 4]]

#Loan Approved



#Use Random Forest for better accuracy


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

