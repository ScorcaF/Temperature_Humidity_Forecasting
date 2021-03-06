
import tensorflow as tf

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
