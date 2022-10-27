import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv('data.csv')
X = data[[str(i) for i in range(69)]]
y = data['cat']

# One-hot encode the labels
o = OneHotEncoder()
y = o.fit_transform(y.values.reshape(-1, 1)).toarray()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
clf = RandomForestClassifier(max_depth=20, n_estimators=500, random_state=0)
clf.fit(X_train, y_train)

print(accuracy_score(y_test, clf.predict(X_test)))
print(f1_score(y_test, clf.predict(X_test), average='macro'))
print(recall_score(y_test, clf.predict(X_test), average='macro'))
print(precision_score(y_test, clf.predict(X_test), average='macro'))

# Save the model
with open('random_forest_classifier_v2.pkl', 'wb') as fid:
    pickle.dump(clf, fid)

# plot confusion matrix
# import matplotlib.pyplot as plt
# import matplotlib
# # matplotlib.use('TkAgg')
# import seaborn as sns
# import numpy as np
#
# cm = confusion_matrix(y_test.argmax(axis=1), clf.predict(X_test).argmax(axis=1))
# cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# plt.figure(figsize=(10, 10))
# sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['bed', 'floor', 'sitting', 'standing'],
#             yticklabels=['bed', 'floor', 'sitting', 'standing'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.savefig('confusion_matrix.png', dpi=500)
# plt.show()



