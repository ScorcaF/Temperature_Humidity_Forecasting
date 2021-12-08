import tensorflow as tf
class CustomEarlyStopping(tf.keras.callbacks.Callback):
  def on_train_begin(self, logs=None):
    self.num_epochs = 0
    
  def on_epoch_end(self, epoch, logs):
    self.num_epochs +=1
    if logs['val_multi_outputMAE'][0] < 0.7 and logs['val_multi_outputMAE'][1] < 2.5 and self.num_epochs >=15:
      print("\nEarly stopping conditions satisfied")
      self.model.stop_training = True
