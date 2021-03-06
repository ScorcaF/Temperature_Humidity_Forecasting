import numpy as np
import tensorflow.lite as tflite
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True)
args = parser.parse_args()
if version == "a":
    output_width = 3
elif version == "a":
    output_width = 9
#Loading test dataset
tensor_specs = (tf.TensorSpec([None, 6, 2], dtype=tf.float32), tf.TensorSpec([None,output_width, 2]))
test_ds = tf.data.experimental.load('./th_test', tensor_specs) 
test_ds = test_ds.unbatch().batch(1)

#Loading tflite model
tflite_model_dir = "Group14_th_" + args.version + ".tflite.zlib"
tflite_compressed_model = open(tflite_model_dir, 'rb').read()
tflite_model = zlib.decompress(tflite_compressed_model)
interpreter = tflite.Interpreter(model_content = tflite_model)


#Preparing for inference
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

#Predict test samples
y_true = []
y_pred = []
for series in test_ds:
    y_true.append(series[1].numpy())
    interpreter.set_tensor(input_details[0]['index'], series[0])
    
    interpreter.invoke()
    
    my_output = interpreter.get_tensor(output_details[0]['index'])
    y_pred.append(my_output[0])

    
    
#Evaluating performance 
temp, hum = multi_outputMAE(y_true, y_pred)
size = os.path.getsize(tflite_model_dir) / 2.**10


#Printing results  
print("Temperature mae on test set: {:0.03}.\nHumidity mae on test set: {:0.03}.\n Model size: {:} kB.".\
      format(temp, hum, size))

#Defining metric
def multi_outputMAE(y_true, y_pred):
    y_true = np.array(y_true).squeeze()
    y_pred = np.array(y_pred).squeeze()

    mae = np.abs(y_true - y_pred)
    mae = np.mean(mae, axis = 1)
    mae = np.mean(mae, axis = 0)
    return mae
