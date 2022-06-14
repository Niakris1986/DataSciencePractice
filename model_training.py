import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

advertising = pd.read_csv('advertising.csv')
advertising.head()
print(advertising.info())
print(advertising.describe())
sns.jointplot(x='Age', y='Area Income', data=advertising)
plt.show()
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=advertising, kind='kde')
plt.show()
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=advertising, color='red')
plt.show()
sns.pairplot(advertising, hue='Clicked on Ad', palette='rocket')
plt.show()
X = advertising[['Age', 'Area Income', 'Daily Internet Usage']]
y = advertising['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=333)
lrm = LogisticRegression()
lrm.fit(X_train, y_train)
predictions = lrm.predict(X_test)
print(classification_report(y_test, predictions))
