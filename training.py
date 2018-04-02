import pandas as pd
import numpy as np
import pickle

from argparse import ArgumentParser
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

parser = ArgumentParser()
parser.add_argument("filename", help="name of trained model file")
data = parser.parse_args()

csv = 'data/' + data.filename + '.csv'

bm = []

cols = ['x', 'y', 'd']

df = pd.read_csv(csv, header=None, names=cols)

X = np.array(df.iloc[:, 0:2])
y = np.array(df.iloc[:, 2])

classifiers = [
    ('kNN', KNeighborsClassifier(n_neighbors=4)),
    ('LR', LogisticRegression()),
    ('SVM', SVC()),
    ('DT', DecisionTreeClassifier())
]

for name, clf in classifiers:
    filename = 'models/%s_%s.pickle' % (name, data.filename)

    print('training: %s' % name)
    rs = np.random.RandomState(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=rs)
    model = clf.fit(X_train, y_train)
    cv = cross_val_score(clf, X_test, y_test, cv=10,
                         scoring='accuracy')
    acc = np.mean(cv)
    predictions = clf.predict(X_test)
    report = classification_report(y_test, predictions)
    print('training %s done... acc= %f' % (name, acc))

    pickle.dump(model, open(filename, 'wb'))

    bm.append('%s %s' % (name, report))

for i in sorted(bm):
    print(i)
