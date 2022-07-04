import neptune.new as neptune
import os

run = neptune.init(
    project=os.environ['NEPTUNE_PROJECT_KEY'],
    api_token=os.environ['NEPTUNE_API_TOKEN'],
    tags=['example']
)

params = {"learning_rate": 0.001, "optimizer": "Adam"}
run["parameters"] = params

for epoch in range(10):
    run["train/loss"].log(0.9 ** epoch)

run["eval/f1_score"] = 0.66

run.stop()