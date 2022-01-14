import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import zlib



class multi_outputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='multi_outputMAE', **kwargs):
        super(multi_outputMAE, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros', shape = (2,))
        self.counter = self.add_weight(name='counter', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight = None):
        values = tf.subtract(y_true, y_pred)
        values = tf.abs(values)
        values = tf.math.reduce_mean(values, axis = 1)
        values = tf.math.reduce_mean(values, axis = 0)
        self.mae.assign_add(values)
        self.counter.assign_add(1)

        
    def reset_states(self, sample_weight = None):
        self.mae.assign(tf.zeros_like(self.mae))
        self.counter.assign(tf.zeros_like(self.counter))


    def result(self):
        return tf.math.divide_no_nan(self.mae, self.counter)
      
      
class WindowGenerator:
    def __init__(self, input_width, output_width, mean, std):
        self.input_width = input_width
        self.output_width = output_width
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-self.output_width, :] 
        labels = features[:, -self.output_width:, :]
        
        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+self.output_width,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
      
class CustomEarlyStopping(tf.keras.callbacks.Callback):
   
  def on_epoch_end(self, epoch, logs):
    if logs['val_multi_outputMAE'][0] < 0.5 and logs['val_multi_outputMAE'][1] < 2.2 and epoch >=15:
      print("\nEarly stopping conditions satisfied")
      self.model.stop_training = True
    
    
    
    
    
    
    
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
    #parameters for structured pruning
    alpha = 1/8
    beta = 1/16
    #Model definition: MLP
    model = tf.keras.Sequential([
                          tf.keras.layers.Flatten(input_shape=(6, 2)),
                          tf.keras.layers.Dense(units=alpha*128, activation = 'relu'),
                          tf.keras.layers.Dense(units=beta*128, activation = 'relu'),
                          tf.keras.layers.Dense(units=2*output_width),
                          tf.keras.layers.Reshape((output_width, 2))])
    
    #Final sparsity of the CNN
    final_sparsity = 0.78
    
elif args.version == "b":
    output_width = 9
    
    #parameters for structured pruning
    alpha = 0.125
    beta = 0.125
    #Model definition: CNN
    model = tf.keras.Sequential([
                          tf.keras.layers.Conv1D(filters=alpha*64, kernel_size=3, activation = "relu"),
                          tf.keras.layers.Flatten(),
                          tf.keras.layers.Dense(units=beta*64, activation = 'relu'),
                          tf.keras.layers.Dense(units=2*output_width),
                          tf.keras.layers.Reshape((output_width, 2))])
    #Final sparsity of the CNN
    final_sparsity = 0.8
    
#Creating windows dataset, to feed to the model
generator = WindowGenerator(input_width, output_width, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

#Saving datasets
tf.data.experimental.save(train_ds, './th_train')
tf.data.experimental.save(val_ds, './th_val')
tf.data.experimental.save(test_ds, './th_test')


callbacks = [CustomEarlyStopping()] 


#Training the model for the first time
input_shape = [32, 6, 2]
model.build(input_shape)
model.compile(optimizer='adam',
            loss = tf.keras.losses.mean_squared_error,
            metrics= [multi_outputMAE()])

model.fit(train_ds, validation_data = val_ds, epochs=20, callbacks=callbacks, verbose = 0)



#Magnitude based pruning parameters
pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.30,    
                                                                           final_sparsity=final_sparsity,
                                                                           begin_step=len(train_ds)*5,
                                                                           end_step=len(train_ds)*15)}
callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), CustomEarlyStopping()] 

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude 
pruned_model = prune_low_magnitude(model, **pruning_params)

# Training the model a second time, to apply magnitude-based pruning
input_shape = [32, 6, 2]
pruned_model.build(input_shape)
pruned_model.compile(optimizer='adam',
            loss = tf.keras.losses.mean_squared_error,
            metrics= [multi_outputMAE()])

pruned_model.fit(train_ds, validation_data = val_ds, callbacks=callbacks, epochs = 20, verbose = 0)

pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
run_model = tf.function(lambda x: pruned_model(x))   

#Saving the model
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
model.save('./model_{}'.format(args.version), signatures=concrete_func)


# Converting and saving the model in a compressed file
tflite_model_dir = "Group14_th_" + args.version + ".tflite.zlib"
converter = tf.lite.TFLiteConverter.from_saved_model('./model_{}')
converter.optimizations = [tf.lite.Optimize.DEFAULT]   #Weight quantization
tflite_model = converter.convert() 
with open(tflite_model_dir, 'wb') as fp:     
    tflite_compressed = zlib.compress(tflite_model)     
    fp.write(tflite_compressed)
