import cupy as np 


class FlattenLayer:
    """
    Takes the volume coming from convolutional & pooling layers. It flattens it and it uses it in the next layers.
    """
    
    def forward_prop(self, z_in, a_in,_):
        self.input_shape = a_in.shape # stored for backprop
        # Flatten the image
        
        featureMaps_z_flattened = z_in.reshape(self.input_shape[0], -1)
        featureMaps_a_flattened = a_in.reshape(self.input_shape[0], -1)
        return featureMaps_z_flattened, featureMaps_a_flattened
    
    def getInfo(self):
        return "Flat"
    
    def backward_prop(self, delta,_):
            return delta.reshape(self.input_shape)