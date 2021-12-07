import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from Temperature_Humidity_Forecasting.WindowGenerator import WindowGenerator 
from Temperature_Humidity_Forecasting.multi_outputMAE import multi_outputMAE 

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#Downloading dataset
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

#Splitting dataset
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6

if args.version == "a":
    output_width = 3
elif args.version == "b":
    output_width = 9

#Definine callback to save best only
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    './model',
    monitor='val_loss',
    verbose=0, 
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch'
)

#Creating windows dataset, to feed to the model
generator = WindowGenerator(input_width, output_width, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

#Model definition
model = tf.keras.Sequential([
                          tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation = "relu"),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(units=64, activation = 'relu'),
                          tf.keras.layers.Dense(units=2*output_width),
                          tf.keras.layers.Reshape((output_width, 2))])])

model.compile(optimizer='adam',
            loss = tf.keras.losses.mean_squared_error,
            metrics= [multi_outputMAE()])

history = model.fit(train_ds, validation_data = val_ds, epochs=20, callbacks=[cp_callback])
