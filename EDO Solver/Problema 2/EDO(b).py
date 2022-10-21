
import keras
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

import matplotlib.pyplot as plt 
import numpy as np 

class SolED(Sequential):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        self.loss_tracker = keras.metrics.Mean(name = 'loss')

    @property
    def metrics(self):
        return [self.loss_tracker]
    
    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x =tf.random.uniform((batch_size, 1), minval = -5, maxval = 5)

        with tf.GradientTape() as tape:
            
            with tf.GradientTape(persistent = True ) as g:
                g.watch(x)
                with tf.GradientTape() as gg:
                    gg.watch(x)
                    y_pred = self(x, training = True)
                
                y_x = gg.gradient(y_pred, x)
            y_xx =g.gradient(y_x,x)
            #Condiciones iniciales.
            x_0 = tf.zeros((batch_size, 1)) 
            y_0 = self(x_0, training = True)

            #Ecuación diferencial a resolver.
            eq = y_xx+y_pred

            ic = y_0-1 # = 0
            #ic2 = y_0 -0.5
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)

        #Para encontrar la derivada. 
        grads = tape.gradient(loss, self.trainable_variables)
        #Actualizar las variables con las gradientes 
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #Observar cual es la función de costo
        self.loss_tracker.update_state(loss)
        #Imprimir la función de costo
        return {'loss': self.loss_tracker.result()}

model =  SolED()
#La función tangente sirve bastante para ajustar funciones analíticas
model.add(Dense(10, activation = 'tanh', input_shape = (1,)))
model.add(Dense(5, activation = 'tanh'))
model.add(Dense(2, activation = 'tanh'))
model.add(Dense(1, activation = 'tanh'))
model.add(Dense(1, activation = 'linear'))

model.summary()

model.compile(optimizer = RMSprop(), metrics = ['loss'])

x = tf.linspace(-5, 5, 300)
history = model.fit(x, epochs = 350, verbose = 1)

#Solución analítica de la ecuación diferencial.
def SA(x, c1, c2):
    return(c1*np.cos(x) + c2*np.sin(x))

x_testv = tf.linspace(-5, 5, 300)
a = model.predict(x_testv)
plt.plot(x_testv, a, label = 'Sol. de la Red')
plt.legend()
plt.plot(x_testv, SA(x,1,0), label = 'Sol. Analítica')
plt.legend()
plt.title('Gráfica   ' + r'$y=Cos(x)-0.5Sin(x)$')
plt.show()
exit() 