
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout,Activation
from tensorflow.keras.optimizers import RMSprop, Adam

from matplotlib import pyplot as plt
import numpy as np
import math


class ODEsolver(Sequential):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mse = tf.keras.losses.MeanSquaredError()

    @property
    def metrics(self):
      return [self.loss_tracker]

    def train_step(self, data):
         batch_size = tf.shape(data)[0]
         batch_size = tf.shape(data)[0]
         xmin, xmax = -5., 5.
         x = tf.random.uniform((batch_size,1), minval=xmin, maxval=xmax)

        with tf.GradientTape() as tape:
            with tf.GradientTape() as tape2:
                tape2.watch(x)

                with tf.GradientTape() as tape3:
                    tape3.watch(x)
                    y_pred = self(x, training=True)
                dy = tape3.gradient(y_pred, x)      # primera derivada
            d2y = tape2.gradient(dy, x)             # segunda derivada

            # Residual de la EDO: y'' + y = 0
            eq = d2y + y_pred

            # Condiciones iniciales
            x0 = tf.zeros((batch_size,1))
            y0 = self(x0, training=True)
            with tf.GradientTape() as tape4:
                tape4.watch(x0)
                y0_pred = self(x0, training=True)
            dy0 = tape4.gradient(y0_pred, x0)

            loss = ( self.mse(eq, 0.)
                   + 10*self.mse(y0, 1.)          # y(0)=1
                   + 10*self.mse(dy0, -0.5) )     # y'(0)=-0.5

        # Apply grads
         grads = tape.gradient(loss, self.trainable_variables)
         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #update metrics
         self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
         return {"loss": self.loss_tracker.result()}
    

model = ODEsolver()

model.add(Dense(128, activation='tanh', input_shape=(1,)))
model.add(Dense(64, activation='tanh'))
model.add(Dense(32, activation='tanh'))

model.add(Dense(1))

model.compile(optimizer=keras.optimizers.Adam(), metrics=['loss'])

dummy = tf.zeros((200,1))
history = model.fit(dummy, epochs=500, verbose=0)

x_test = np.linspace(-5,5,100).reshape(-1,1)
y_pred = model.predict(x_test)

# Solución exacta: y(x)=cos(x) -0.5 sin(x)
y_exact = np.cos(x_test) - 0.5*np.sin(x_test)

plt.plot(x_test, y_pred, label="aprox")
plt.plot(x_test, y_exact, label="Exact", linestyle="dashed")
plt.legend()
plt.show()
