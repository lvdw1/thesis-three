from processor import *
from transformer import *
from utils import read_csv_data

data = "runs/training/final/track10.csv"
track = "sim/tracks/track10.json"


processor = Processor()

track_data = processor.build_track_data(track)
data = read_csv_data(data)
processed_data = processor.process_csv(data, track_data)
print(len(processed_data[0]))

transformer = FeatureTransformer()
df_trans = transformer.fit_transform(
    processed_data,
    exclude_cols=["steering", "throttle", "brake"],
    pca_variance=0.99,
)

print(len(df_trans[0]))

