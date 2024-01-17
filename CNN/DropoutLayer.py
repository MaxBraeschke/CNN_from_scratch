import cupy as np

class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward_prop(self, z_in, a_in, training):
        if training:
            self.mask = np.random.rand(*a_in.shape) >= self.dropout_rate
            a_out = a_in * self.mask / (1 - self.dropout_rate)
            return z_in, a_out
        else:
            return z_in, a_in
        
    def getInfo(self):
        return "DL{}".format(self.dropout_rate)

    def backward_prop(self, delta,_):
        return delta * self.mask / (1 - self.dropout_rate)
