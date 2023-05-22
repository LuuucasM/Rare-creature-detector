import cv2, time
from mss import mss
from PIL import Image
import numpy as np
import pickle
import random
import copy

class FullConC():

    def __init__(self, name, load, num_layers=0, layer_size=0, learning_rate=0, momentum=0, alpha=0):
        if load == False:
            self._name = str(name)
            self._num_layers = num_layers
            self._layer_size = layer_size
            self._learning_rate = learning_rate
            self._alpha = alpha
            self._momentum = momentum
            self._learning_rate = learning_rate
            self._inputs_size = 2600 #need to adjust manually I think I use 3 right now
            self._output_size = 256
            #creating the matrixes to represent each part of the neural network
            #-----input weights
            self._weight_ih = np.random.uniform(low=-1, high=1, size=(self._inputs_size, self._layer_size))*np.sqrt(1/self._inputs_size)
            self._bias_ih = np.random.uniform(low=-1, high=1, size=(self._layer_size, 1))*np.sqrt(1/self._layer_size)
            self._bias_dho = np.random.uniform(low=-1, high=1, size=(self._inputs_size, 1))*np.sqrt(1/self._inputs_size)
            
            #-----hidden weights
            self._weight_hh = [None]*(self._num_layers-1)
            self._bias_hh = [None]*(self._num_layers-1)
            self._bias_dhh = [None]*(self._num_layers-1)
            for i in range(self._num_layers-1):
                self._weight_hh[i] = np.random.uniform(low=-1, high=1, size=(self._layer_size, self._layer_size))*np.sqrt(1/self._layer_size)
                self._bias_hh[i] = np.random.uniform(low=-1, high=1, size=(self._layer_size, 1))*np.sqrt(1/self._layer_size)
                self._bias_dhh[i] = np.random.uniform(low=-1, high=1, size=(self._layer_size, 1))*np.sqrt(1/self._layer_size)

            #-----output weights
            self._weight_ho = np.random.uniform(low=-1, high=1, size=(self._layer_size, self._output_size))*np.sqrt(1/self._layer_size)
            self._bias_ho = np.random.uniform(low=-1, high=1, size=(self._output_size, 1))*np.sqrt(1/self._output_size)
            self._bias_dih = np.random.uniform(low=-1, high=1, size=(self._layer_size, 1))*np.sqrt(1/self._layer_size)
        
        else:
            with open(str(name)+"_fcnn.pickle", "rb") as f:
                self.__dict__.clear()
                self.__dict__.update(pickle.load(f))

    def forward_pass(self, x):
        ##ENCODER
        #-----input to hidden
        z_ih = np.dot(self._weight_ih.transpose(), x) + self._bias_ih
        activation_ih = self.Relu(z_ih)
        activation_last = activation_ih
        
        #-----hidden to hidden
        z_hh = [None]*(self._num_layers-1)
        activation_hh = [None]*(self._num_layers-1)
        for i in range(self._num_layers-1):
            z_hh[i] = np.dot(self._weight_hh[i].transpose(), activation_last) + self._bias_hh[i]
            activation_hh[i] = self.Relu(z_hh[i])
            activation_last = activation_hh[i]

        #-----hidden to output
        z_ho = np.dot(self._weight_ho.transpose(), activation_last) + self._bias_ho
        activation_ho = self.Relu(z_ho)
        activation_last = activation_ho

        ##DECODER

        #-----output to hidden
        z_dih = np.dot(self._weight_ho, activation_last) + self._bias_dih
        activation_dih = self.Tanh(z_dih)
        activation_last = activation_dih

        #-----hidden to hidden
        z_dhh = [None]*(self._num_layers-1)
        activation_dhh = [None]*(self._num_layers-1)
        for i in range(self._num_layers-2, -1, -1):
            z_dhh[i] = np.dot(self._weight_hh[i], activation_last) + self._bias_dhh[i]
            activation_dhh[i] = self.Tanh(z_dhh[i])
            activation_last = activation_dhh[i]

        #----- hidden to input
        z_dho = np.dot(self._weight_ih, activation_last) + self._bias_dho
        activation_dho = self.Sigmoid(z_dho)
        

        return z_ih, z_hh, z_ho, activation_ih, activation_hh, activation_ho, z_dih, z_dhh, z_dho, activation_dih, activation_dhh, activation_dho

    def learn(self, x):
        #forward pass
        x = x/255
        z_ih, z_hh, z_ho, activation_ih, activation_hh, activation_ho, z_dih, z_dhh, z_dho, activation_dih, activation_dhh, activation_dho = self.forward_pass(x)

        #loss
        loss = -np.sum((x*np.log(activation_dho)) + ((1-x)*np.log(1-activation_dho)))


        #backward pass ##DECODER
        #-----inpput error
        error_out = (activation_dho - x)
        cost_w_dho = np.dot(activation_dhh[0], error_out.transpose())
        cost_b_dho = error_out
        error_last = error_out
        weight_last = self._weight_ih

        #-----hidden errors
        error_dhh = [None]*(self._num_layers-1)
        cost_w_dhh = [None]*(self._num_layers-1)
        cost_b_dhh = [None]*(self._num_layers-1)
        for i in range(0, self._num_layers-2, 1):
            error_dhh[i] = np.dot(weight_last.transpose(), error_last)*self.Tanh_prime(z_dhh[i])
            cost_w_dhh[i] = np.dot(activation_dhh[i+1], error_dhh[i].transpose())
            cost_b_dhh[i] = error_dhh[i]
            error_last = error_dhh[i]
            weight_last = self._weight_hh[i]

        error_dhh[-1] = np.dot(weight_last.transpose(), error_last)*self.Tanh_prime(z_dhh[-1])
        cost_w_dhh[-1] = np.dot(activation_dhh[-1], error_dhh[-1].transpose())
        cost_b_dhh[-1] = error_dhh[-1]
        error_last = error_dhh[-1]
        weight_last = self._weight_hh[-1]
            
        #-----output errors
        error_dih = np.dot(weight_last.transpose(), error_last)*self.Tanh_prime(z_dih)
        cost_w_dih = np.dot(activation_ho, error_dih.transpose())
        cost_b_dih = error_dih
        error_last = error_dih
        weight_last = self._weight_ho


        #backward pass ##ENCODER
        #-----output error
        error_out = np.dot(weight_last.transpose(), error_last)*self.Relu_prime(z_ho)
        cost_w_ho = np.dot(activation_hh[-1], error_out.transpose())
        cost_b_ho = error_out
        error_last = error_out
        weight_last = self._weight_ho
        
        #-----hidden errors
        error_hh = [None]*(self._num_layers-1)
        cost_w_hh = [None]*(self._num_layers-1)
        cost_b_hh = [None]*(self._num_layers-1)
        for i in range(self._num_layers-2, 0, -1):
            error_hh[i] = np.dot(weight_last, error_last)*self.Relu_prime(z_hh[i])
            cost_w_hh[i] = np.dot(activation_hh[i-1], error_hh[i].transpose())
            cost_b_hh[i] = error_hh[i]
            error_last = error_hh[i]
            weight_last = self._weight_hh[i]

        error_hh[0] = np.dot(weight_last, error_last)*self.Relu_prime(z_hh[0])
        cost_w_hh[0] = np.dot(activation_ih, error_hh[0].transpose())
        cost_b_hh[0] = error_hh[0]
        error_last = error_hh[0]
        weight_last = self._weight_hh[0]
        
        #-----input errors
        error_ih = np.dot(weight_last, error_last)*self.Relu_prime(z_ih)
        cost_w_ih = np.dot(x, error_ih.transpose())
        cost_b_ih = error_ih
        error_last = error_ih
        weight_last = self._weight_ih

        #update each weight
        #-----input weights
        self._weight_ih += -self._learning_rate*cost_w_ih - (self._learning_rate*(cost_w_dho.transpose()))
        self._bias_ih += -self._learning_rate*cost_b_ih
        self._bias_dho += -self._learning_rate*cost_b_dho

        #-----hidden weights
        for i in range(self._num_layers-1):
            self._weight_hh[i] += -self._learning_rate*cost_w_hh[i] - (self._learning_rate*(cost_w_dhh[i].transpose()))
            self._bias_hh[i] += -self._learning_rate*cost_b_hh[i]
            self._bias_dhh[i] +=-self._learning_rate*cost_b_dhh[i]

        #-----output weights
        self._weight_ho += -self._learning_rate*cost_w_ho - (self._learning_rate*(cost_w_dih.transpose()))
        self._bias_ho += -self._learning_rate*cost_b_ho
        self._bias_dih += -self._learning_rate*cost_b_dih

        
        return loss, error_last, weight_last

    def predict(self, x):
        #forward pass
        x = x/255
        z_ih, z_hh, z_ho, activation_ih, activation_hh, activation_ho, z_dih, z_dhh, z_dho, activation_dih, activation_dhh, activation_dho = self.forward_pass(x)
        loss = -np.sum((x*np.log(activation_dho)) + ((1-x)*np.log(1-activation_dho)))
        loss = np.divide(np.square(loss), 100000)
        return loss

    def Relu(self, x):
        return x * (x > 0)

    def Relu_prime(self, x):
        return (x > 0)

    def save(self):
        with open(self._name+"_fcnn.pickle", "wb") as f:
            pickle.dump(self.__dict__, f, 2)
    
    def Softmax(self, x):
        pred = np.exp(x)/np.sum(np.exp(x))
        return pred

    def Tanh(self, x):
        return np.tanh(x)

    def Tanh_prime(self, x):
        return 1 - np.square(np.tanh(x))
    
    def Sigmoid(self, x):
        return 1/(1+np.exp(-x))

def main():
    ## generating samples
    '''
    try:
        with open("no_rare.pkl", "rb") as f:
            lst = pickle.load(f)
    except FileNotFoundError:
        lst = []
    sct = mss()
    #second monitor full screen
    num_1 = np.array(sct.grab({'top': 410, 'left': 3605, 'width':130, 'height': 20}))
    num_1 = cv2.cvtColor(num_1, cv2.COLOR_BGRA2GRAY)
    cv2.imshow("Capture", num_1)
    cv2.waitKey(30)
    cv2.destroyAllWindows()
    lst.append(num_1)
    with open("no_rare.pkl", "wb") as f:
        pickle.dump(lst, f)
    '''
    ## training network
    '''
    loss = 0
    network = FullConC("no_rare", False, learning_rate=0.001, momentum = 0.9, alpha=0, num_layers=3, layer_size=1200)
    try:
        with open("no_rare.pkl", "rb") as f:
            lst = pickle.load(f)
    except FileNotFoundError:
        print("couldnt find pics")
        exit()
    for i in range((len(lst)*2)):
        temp = np.random.randint(len(lst))
        loss += network.learn(lst[temp].reshape(-1, 1))[0]
        if i%100 == 0:
            loss = loss/100
            print("iteration %d with %f average loss over 100 points" % (i, loss))
            loss = 0
    network.save()
    '''
    ## prediction
    try:
        with open("no_rare.pkl", "rb") as f:
            lst = pickle.load(f)
    except FileNotFoundError:
        print("cant find pic file")
        exit()
    sct = mss()
    network = FullConC("no_rare", True, learning_rate=0.001, momentum = 0.9, alpha=0, num_layers=3, layer_size=1200)
    while True:
        num_1 = np.array(sct.grab({'top': 410, 'left': 3605, 'width':130, 'height': 20}))
        num_1 = cv2.cvtColor(num_1, cv2.COLOR_BGRA2GRAY)
        #cv2.imshow("Capture", num_1)
        #cv2.waitKey(1000)
        prediction = network.predict(num_1.reshape(-1,1))
        if prediction > 23:
            print("rare creat detec")


        time.sleep(1)

    '''
            lst.append(num_1)
            print("added new sample")
            with open("no_rare.pkl", "wb") as f:
                pickle.dump(lst, f)
    '''
if __name__ == "__main__":
    main()
