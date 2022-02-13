import pandas as pd
import numpy as np

training_df = pd.read_csv(filepath_or_buffer="Banknotes.csv")
print(training_df.size)
float_col = training_df.select_dtypes(include=['float64'])

for col in float_col.columns.values:
    training_df[col] = np.floor(pd.to_numeric(training_df[col], errors='coerce')).astype('float32')

my_feature = "Varience"  # the total number of rooms on a specific city block.
my_label = "Class"  # the median value of a house on a specific city block.

x = training_df[my_feature]
y = training_df[my_label]

print(y)
print(x)
