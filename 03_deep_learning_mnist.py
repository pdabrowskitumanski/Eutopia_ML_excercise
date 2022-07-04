from tensorflow.keras.datasets import mnist
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
    tags=['deep_learning_mnist']
)
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')


# dataset build
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# logging some images
for image in train_images[:100]:
    run['test/sample_images'].log(neptune.types.File.as_image(image))

# reshaping
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# model
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
params = {
    'epochs': 5,
    'batch_size': 128,
}

model.fit(train_images, train_labels, callbacks=[neptune_cbk], **params)

# logging data
run['hyper-parameters'] = params
run.stop()
