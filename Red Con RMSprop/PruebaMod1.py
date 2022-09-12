
import networkMod 
import pickle 
import mnist_loader 


DTraining, DValidation, DTest = mnist_loader.load_data_wrapper()

DTraining = list(DTraining)
DTest = list(DTest)

"""Prueba de la red neuronal básica implementando el
optimizador RMSprop"""
net = networkMod.Network([784, 30, 10])
net.RMSprop(DTraining, 30, 10, 3, .1, .2, test_data = DTest)

File = open('red_prueba.pkl','wb')
pickle.dump(net, File)
File.close()
exit()