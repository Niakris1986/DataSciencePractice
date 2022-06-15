import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

loan_data = pd.read_csv('loan_data.csv')
print(loan_data.head())
print(loan_data.describe())
print(loan_data.info())
plt.figure(figsize=(10, 6))
loan_data[loan_data['credit.policy'] == 0]['fico'].hist(alpha=0.5,color='green',
                                              bins=30, label='Credit.Policy = 0')
loan_data[loan_data['credit.policy'] == 1]['fico'].hist(bins=35, color='blue',
                                                        alpha=0.6, label='credit.policy = 1')
plt.legend()
plt.xlabel('FICO')
plt.show()
plt.figure(figsize=(11, 7))
sns.countplot(x='purpose', hue='not.fully.paid', data=loan_data, palette='Set3')
plt.show()
sns.jointplot(x='fico', y='int.rate', data=loan_data, color='green')
plt.show()
sns.lmplot(x='fico', y='int.rate', data=loan_data, hue='credit.policy', col='not.fully.paid', palette='Set2')
plt.show()
print(loan_data.info())
cat_feats = ['purpose']
final_data = pd.get_dummies(loan_data, columns=cat_feats, drop_first=True)
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
pred = dtree.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfc.pred = rfc.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


