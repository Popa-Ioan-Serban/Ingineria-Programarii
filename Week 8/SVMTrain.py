from os import path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

bankdata = pd.read_csv('dataset.csv')
attributes = bankdata.drop('Class', axis=1)
labels = bankdata['Class']
X_train, X_test, y_train, y_test = train_test_split(attributes, labels, test_size=0.05)

if path.exists('model.joblib'):
    svclassifier = load('model.joblib')
    print('Model LOADED successfully!')
else:
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    dump(svclassifier, 'model.joblib')
    print('Model SAVED successfully!')

y_pred = svclassifier.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(y_test)
print(y_pred)
