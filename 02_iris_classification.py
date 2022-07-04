from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import neptune.new as neptune
import neptune.new.integrations.sklearn as npt_utils
import plotly.express as px


run = neptune.init(
    project=os.environ['NEPTUNE_PROJECT_KEY'],
    api_token=os.environ['NEPTUNE_API_TOKEN'],
    tags=['iris_classification']
)

# loading dataset
iris = sns.load_dataset('iris')
df = pd.DataFrame(iris)
X = df.drop(['species'], axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=5)


# defining the model - change according to your need!
parameters = {'n_neighbors': 2}
classifier = KNeighborsClassifier(**parameters)
run['parameters'] = parameters

# fitting
classifier.fit(X_train, y_train)

# predictions
predictions = classifier.predict(X_test)
run['scores/accuracy'] = metrics.accuracy_score(y_test, predictions)

# logging additional data
run['cls_summary'] = npt_utils.create_classifier_summary(classifier, X_train, X_test, y_train, y_test)


# stop logging the data
run.stop()
