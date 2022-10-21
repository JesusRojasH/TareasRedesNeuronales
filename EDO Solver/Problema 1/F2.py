
import numpy as np
import matplotlib.pyplot as plt 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler

x= np.arange(-1,1,0.01)

def f(x):
    return(1 + 2*x + 4*x**3)

x = x.reshape((len(x), 1))
fun = f(x).reshape((len(f(x)), 1))


"""La ecuación MinMaxScaler normaliza los datos meiante la diferencia 
entre el valor máximo y el mínimo."""
scale_x = MinMaxScaler() 
x = scale_x.fit_transform(x)

F_norm = MinMaxScaler()
fun = F_norm.fit_transform(fun)


model = Sequential()
model.add(Dense(10, activation='tanh', input_dim=1))
model.add(Dense(10, activation='tanh'))
model.add(Dense(1, activation = 'linear'))
model.summary()

model.compile(loss='mse',optimizer=RMSprop(learning_rate=0.01), metrics=['accuracy'])
history = model.fit(x, f(x), batch_size=10,epochs=40,verbose=1)

y_predN = model.predict(x)

#Regresar las variables a su estado original 
x_plot = scale_x.inverse_transform(x)
y = F_norm.inverse_transform(f(x))
y_pred = F_norm.inverse_transform(y_predN)

plt.plot(x_plot,y, label='Solución Analítica')

plt.plot(x_plot, y_pred, label='Solución Red Neuronal')
plt.title(r'$y(x)=1+2x+4x^3$')
plt.legend()
plt.show()