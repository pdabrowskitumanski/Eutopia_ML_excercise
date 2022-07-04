import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import neptune.new as neptune
import os
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

"""
This project comes from the book Deep Learning with Python (second edition) by F. Chollet
"""

run = neptune.init(
    project=os.environ['NEPTUNE_PROJECT_KEY'],
    api_token=os.environ['NEPTUNE_API_TOKEN'],
    tags=['deep_learning_iris']
)
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

# loading data
iris = sns.load_dataset('iris')
df = pd.DataFrame(iris)

# reshaping the data
lbl_clf = LabelEncoder()
Y = df['species']
X = df.drop(['species'], axis=1)
Y_encoded = lbl_clf.fit_transform(Y)
Y_final = keras.utils.to_categorical(Y_encoded)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(X, Y_final, test_size=0.25,
                                                    random_state=0, stratify=Y_encoded, shuffle=True)

# standarizing the dataset
std_clf = StandardScaler()
x_train_new = std_clf.fit_transform(x_train)
x_test_new = std_clf.transform(x_test)

# model definition
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(3, activation="softmax")
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model fitting
params = {
    'epochs': 50,
    'batch_size': 7
}
iris_model = model.fit(x_train_new, y_train, callbacks=[neptune_cbk], **params)

# logging data
run['hyper-parameters'] = params
run.stop()