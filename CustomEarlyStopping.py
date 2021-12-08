import tensorflow as tf
class CustomEarlyStopping(tf.keras.callbacks.Callback):
  self.num_epochs = 0
  def on_epoch_end(self, epoch, logs):
    self.num_epochs +=1
    if logs['val_multi_outputMAE'] < 1.5 and self.num_epochs >=15:
      print("\nEarly stopping conditions satisfied")
      self.model.stop_training = True
