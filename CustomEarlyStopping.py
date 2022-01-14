import tensorflow as tf
class CustomEarlyStopping(tf.keras.callbacks.Callback):
   
  def on_epoch_end(self, epoch, logs):
    if logs['val_multi_outputMAE'][0] < 0.5 and logs['val_multi_outputMAE'][1] < 2.2 and epoch >=15:
      print("\nEarly stopping conditions satisfied")
      self.model.stop_training = True
