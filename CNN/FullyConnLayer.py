from math import sqrt
import cupy as np

class FullyConnLayer:

    
    def xavier_init_(self, input_size, output_size):
        k = 1.0 / (input_size*output_size)
        weight = np.random.uniform(-np.sqrt(k),np.sqrt(k), (input_size, output_size))
        bias = np.random.uniform(-np.sqrt(k),np.sqrt(k), output_size) 
        return weight, bias
    
    def he_init_(self, input_size, output_size):
        bias = np.random.randn(output_size) * sqrt(2.0/(output_size))
        weight = np.random.randn(input_size, output_size) * sqrt(2.0/(input_size))
        return weight, bias

    def __init__(self, input_size, output_size, activation = "linear", initMethod = "Xavier"):
        self.output_size = output_size
        self.input_size = input_size
        
        if activation == "reLU":
            self.activation = self.reLU
            self.activation_derivative = self.reLU_Derivative
        elif activation == "softmax":
            self.activation = self.softmax
            self.activation_derivative = self.softmax_Derivative
        elif activation == "linear":
            self.activation = self.lin_activation
            self.activation_derivative = self.lin_activation_Derivative

        if initMethod == "Xavier":
            self.weight, self.bias = self.xavier_init_(input_size, output_size)
        elif initMethod == "He":
            self.weight, self.bias = self.he_init_(input_size, output_size)
        


    def getInfo(self):
        return "FC{}".format(self.output_size)
    
    def reLU(self,input):
        return np.where(input > 0,input, 0)
    
    def reLU_Derivative(self,z):
        return z > 0 
        
    
    def lin_activation(self,input):
        return input
    
    
    def lin_activation_Derivative(self,z):
        return 1
    
    
    def softmax(self,input):
        exps = np.exp(input - np.max(input, axis=-1).reshape(input.shape[0],1))
        return exps / np.sum(exps, axis=-1).reshape(input.shape[0],1)
    
    
    def softmax_Derivative(self,z):
        s_d = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        for i in range(z.shape[0]):
            z_s = self.softmax(np.array([z[i]]))[0]
            S_vector = z_s.reshape(z_s.shape[0],1)
            S_matrix = np.tile(S_vector, z_s.shape[0])
            s_d[i] = np.diag(z_s) - (S_matrix* S_matrix.T)

        return s_d



    def forward_prop(self, z_in, a_in,_):
        self.a_in = a_in
        self.z_in = z_in
        self.batch_size = a_in.shape[0]
        
        weights = self.weight
        bias = self.bias

        z_out = np.dot(self.a_in, weights) + bias
        a_out = self.activation(z_out)

        self.a_out = a_out
        self.z_out = z_out
        return z_out, a_out

    def backward_prop(self, delta, eta):

        if(self.activation == self.softmax):
            delta = delta.reshape(32,1,10)@self.activation_derivative(self.z_out)
            delta =delta.reshape(32,10)
        else:
            delta = delta * self.activation_derivative(self.z_out)

        dE_dw = np.zeros((self.batch_size, self.input_size, self.output_size))
        dE_db = np.zeros((self.batch_size, self.output_size))

        #bias derivitve
        dE_db = delta
    
        #weight derivitve
        dE_dw = delta.reshape(self.batch_size, self.output_size, 1) * self.a_in.reshape(self.batch_size,1,self.input_size)
        dE_dw = dE_dw.transpose(0,2,1)

        nabla_b = dE_db.T.sum(axis=-1)
        nabla_w = dE_dw.T.sum(axis=-1).T

        #new delta
        delta = (self.weight@delta.T).T

        #update weights and bias
        self.weight -= (eta * (nabla_w/self.batch_size))
        self.bias -= (eta * (nabla_b/self.batch_size))

        return delta
    
