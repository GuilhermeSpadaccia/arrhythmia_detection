import numpy as np

class neural_network():
    
    def __init__(self, neurons):
        self.num_inputs = None
        self.num_outputs = None
        self.w1 = None
        self.w2 = None
        self.activation = None
        self.neurons = neurons
        self.bias = 1
        self.all_errors = []
        
        
    def startWeights(self, num_inputs, neurons, num_outputs):
        w1 = np.random.uniform(low=0., high=1., size=(num_inputs, neurons))
        w2 = np.random.uniform(low=0., high=1., size=(neurons, num_outputs))
        return w1, w2
    
    
    def tangent(self, values, deriv=False):
        if deriv:
            return 1 - self.tangent(values)**2
        else:
            return (np.exp(values) - np.exp(-values)) / (np.exp(values) + np.exp(-values))
    
    
    def sigmoid(self, values, deriv=False):
        if deriv:
            return (1 - self.sigmoid(values)) * self.sigmoid(values)
        else:
            return 1 / (1 + np.exp(-values))
        
        
    def linear(self, values, deriv=False):
        if deriv:
            return np.ones(len(values))
        else:
            return values
    
    
    def softmax(self, values, deriv=False):
        return np.exp(values)/sum(np.exp(values))
    
    
    def activation_function(self, values, deriv=False, activation='ERRO'):
        if activation == "sigmoid":
            return self.sigmoid(values, deriv)
        elif activation == "tangent":
            return self.tangent(values, deriv)
        elif activation == "linear":
            return self.linear(values, deriv)
        elif activation == "softmax":
            return self.softmax(values, deriv)
        else:
            raise Exception("ACTIVATION FUNCTION ERROR")
    
    
    def feedforward(self, inputs, weights, activation):
        return self.activation_function(np.dot(inputs, weights) + self.bias, False, activation)
    
    
    def calc_error(self, net_output, expect_output):
        return expect_output - net_output
    
    
    def mse_loss_function(self, error):
        return np.array(error**2).mean()
    
    
    def cross_entropy_loss_function(self, nn_output, expect_output):
        
        res = -np.dot(expect_output, np.log(nn_output))
        return res
    
    
    def add_noize(self, values, max_percent=.1):
        for i in range(len(values)):
            delta = np.random.uniform(low=0., high=max_percent)
            sum_sub = round(np.random.uniform(low=0., high=1.))
            
            if sum_sub == 1:
                values[i] += values[i]*delta
            else:
                values[i] -= values[i]*delta
        
        return values
    
    
    def train_validation_error(self, validation_data, validation_target, entropy=False):
        result = self.predict(validation_data)
        if entropy:
            entrps = []
            for i in range(len(result)):
                entrps.append(self.cross_entropy_loss_function(result[i], validation_target[i]))
            return np.array(entrps).mean()
        else:
            return ((np.array(validation_target) - np.array(result))**2).mean()
    
    
    def fit(self, inputs, outputs, learning_rate=0.03, activation=["sigmoid","sigmoid"], epochs=300, validation_data=[], validation_target=[], correction="batch"):
        
        self.activation = activation
        # define the number of neurons on input network
        self.num_inputs = inputs.shape[1]
        # define the number of neurons on output network
        self.num_outputs = outputs.shape[1]
        
        # defino os pesos da rede
        self.w1, self.w2 = self.startWeights(self.num_inputs, self.neurons, self.num_outputs)
        
        train_error = []
        validation_error = []
        
        while epochs > 0:
            print("Epoch: ", epochs)
            
            delta_w1_mean = 0
            delta_w2_mean = 0
            count = 0
            
            for i in range(len(inputs)):
                # feedforward for hiden network
                z = self.feedforward(inputs[i], self.w1, self.activation[0])
                
                # feedforward for output layer
                y = self.feedforward(z, self.w2, self.activation[1])
                
                if self.activation[1] == "softmax":
                    entropy_valid = True
                    
                    # The result of the derivativ is multiplied pra softmax error
                    self.hiden_delta = outputs[i] - y
                    
                    # use the rule of cross-etropy as loss function
                    entropy = self.cross_entropy_loss_function(y, outputs[i])    
                    self.all_errors.append(entropy)
                else:
                    entropy_valid = False
                    
                    error = self.calc_error(y, outputs[i])
                
                    # calculate the mean square error
                    self.mse = self.mse_loss_function(error)
                
                    # Add all MSE on a list
                    self.all_errors.append(self.mse)

                    #------------------
                    # BACKPROPAGATION
                    #------------------
                    self.hiden_delta = self.activation_function(y, True, self.activation[1]) * error
                
                self.delta_w2 = np.dot(z[np.newaxis].T, self.hiden_delta[np.newaxis])
                self.input_delta = np.dot(self.hiden_delta, self.w2.T)
                self.input_delta = self.input_delta * self.activation_function(z, True, self.activation[0])
                self.delta_w1 = np.dot(inputs[i][np.newaxis].T, self.input_delta[np.newaxis])
                
                if correction == "batch":
                    delta_w1_mean += self.delta_w1
                    delta_w2_mean += self.delta_w2
                    count += 1
                else:
                    self.w1 += (self.delta_w1 * learning_rate)
                    self.w2 += (self.delta_w2 * learning_rate)
                    
            if correction == "batch":
                delta_w1_mean = delta_w1_mean/count
                delta_w2_mean = delta_w2_mean/count

                self.w1 += (delta_w1_mean * learning_rate)
                self.w2 += (delta_w2_mean * learning_rate)
            
            tot_train_error = self.train_validation_error(inputs,
                                                          outputs,
                                                          entropy=entropy_valid)
            
            tot_valid_error = self.train_validation_error(validation_data,
                                                          validation_target,
                                                          entropy=entropy_valid)
            
            train_error.append(tot_train_error)
            validation_error.append(tot_valid_error)
            
            print("Train error: ",tot_train_error)
            print("Validation error: ",tot_valid_error)
            
            self.all_errors = []
            epochs -= 1
        
        return train_error, validation_error
        
    def predict(self, inputs):
        outputs = []
        
        for i in range(len(inputs)):
            y = self.feedforward(inputs[i], self.w1, self.activation[0])
            
            z = self.feedforward(y, self.w2, self.activation[1])
            
            outputs.append(z)
    
        return outputs
