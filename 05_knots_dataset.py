import pandas as pd
import seaborn as sns
import neptune.new as neptune
import os

# Neptune initiation
run = neptune.init(
    project=os.environ['NEPTUNE_PROJECT_KEY'],
    api_token=os.environ['NEPTUNE_API_TOKEN'],
    tags=['iris_classification']
)

# loading dataset
df = pd.read_csv('knots_dataset.csv')
df['writhe'] = df.apply(lambda row: abs(row['writhe']), axis=1)
df = df[['writhe', 'crossings', 'topology']]
run['data/df_raw_summary'].upload(neptune.types.File.as_html(df.describe()))
df = df.loc[df['topology'].isin(['0_1', '3_1', '4_1'])]
run['data/df_reduced_summary'].upload(neptune.types.File.as_html(df.describe()))


"""
Place for your code here :)
"""

run.stop()
