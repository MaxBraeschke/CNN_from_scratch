from math import sqrt
import cupy as np

#Zero padding
#Stride of 1 (full convolution)

class ConvLayer:
    """
    Convolutional Layer class for a Convolutional Neural Network (CNN).
    """

    def xavier_init_(self, fM_n_in, fM_n_out, kernelSize):
        """
        Xavier initialization for the convolutional layer.

        Args:
            fM_n_in (int): Number of input feature maps.
            fM_n_out (int): Number of output feature maps.
            kernelSize (int): Size of the kernel.

        Returns:
            tuple: Tuple containing the initialized kernels and bias.
        """
        k = 1.0 / (fM_n_in * kernelSize * kernelSize)
        kernels = np.random.uniform(-np.sqrt(k), np.sqrt(k), (fM_n_out, fM_n_in, kernelSize, kernelSize))
        bias = np.random.uniform(-np.sqrt(k), np.sqrt(k), fM_n_out)
        return kernels, bias

    def he_init_(self, fM_n_in, fM_n_out, kernelSize):
        """
        He initialization for the convolutional layer.

        Args:
            fM_n_in (int): Number of input feature maps.
            fM_n_out (int): Number of output feature maps.
            kernelSize (int): Size of the kernel.

        Returns:
            tuple: Tuple containing the initialized kernels and bias.
        """
        kernels = np.random.randn(fM_n_out, fM_n_in, kernelSize, kernelSize) * np.sqrt(2.0 / (fM_n_in * kernelSize * kernelSize))
        bias = np.random.randn(fM_n_out) * np.sqrt(2.0 / (fM_n_out))
        return kernels, bias

    def __init__(self, inputSize, fM_n_in, fM_n_out, kernelSize, initMethod="Xavier"):
        """
        Initialize the ConvLayer object.

        Args:
            inputSize (int): Size of the input.
            fM_n_in (int): Number of input feature maps.
            fM_n_out (int): Number of output feature maps.
            kernelSize (int): Size of the kernel.
            initMethod (str, optional): Initialization method for the kernels and bias. Defaults to "Xavier".
        """
        self.featureMaps = fM_n_out
        self.kernelSize = kernelSize

        if initMethod == "Xavier":
            self.kernels, self.bias = self.xavier_init_(fM_n_in, fM_n_out, kernelSize)
        elif initMethod == "He":
            self.kernels, self.bias = self.he_init_(fM_n_in, fM_n_out, kernelSize)

        self.pad = int(self.kernelSize / 2)

        self.img_din, self.img_h, self.img_w = fM_n_in, inputSize, inputSize
        self.ker_f_out, self.ker_f_in, self.ker_h, self.ker_w = self.kernels.shape
        self.stride = 1

    def getInfo(self):
        """
        Get information about the ConvLayer.

        Returns:
            str: Information about the ConvLayer.
        """
        return "C{}".format(self.featureMaps)


    def reLU(self, z):
        """
        Apply the ReLU activation function to the given input.

        Args:
            z (ndarray): Input to the ReLU function.

        Returns:
            ndarray: Output after applying the ReLU function.
        """
        return np.where(z > 0.0, z, 0.0)

    def reLU_Derivative(self, z):
        """
        Compute the derivative of the ReLU function for the given input.

        Args:
            z (ndarray): Input to the ReLU derivative function.

        Returns:
            ndarray: Derivative of the ReLU function.
        """
        return np.where(z > 0.0, 1, 0.0)

    def forward_prop(self, z_in, a_in, _):
        """
        Perform forward propagation for the ConvLayer.

        Args:
            z_in (ndarray): Input to the ConvLayer.
            a_in (ndarray): Activation input to the ConvLayer.
            _ (None): Placeholder argument.

        Returns:
            tuple: Tuple containing the output of the ConvLayer.
        """
        self.a_in = a_in
        self.z_in = z_in
        self.batchSize = a_in.shape[0]
        img_din, img_h, img_w = self.img_din, self.img_h, self.img_w
        ker_f_out, ker_f_in, ker_h, ker_w = self.ker_f_out, self.ker_f_in, self.ker_h, self.ker_w
        pad = self.pad
        stride = self.stride

        out_h = int((img_h - ker_h) + (2 * pad) / stride) + 1
        out_w = int((img_w - ker_w) + (2 * pad) / stride) + 1

        #padding the image
        pad_img = np.pad(a_in, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

        self.pad_img = pad_img


        #create indices for the convolution
        i0 = np.repeat(np.arange(ker_h), ker_h)
        i1 = np.repeat(np.arange(img_h), img_h)
        j0 = np.tile(np.arange(ker_w), ker_h)
        j1 = np.tile(np.arange(img_h), img_w)

        self.i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        self.j = j0.reshape(-1, 1) + j1.reshape(1, -1)


        #img_kernels = matrix version of input image
        img_kernels = pad_img[:, :, self.i, self.j]
        
        img_kernels = img_kernels.reshape((self.batchSize, ker_h * ker_w * ker_f_in, out_h * out_w))
        self.img_kernels = img_kernels

        weights = self.kernels.reshape((ker_f_out, ker_h * ker_w * ker_f_in))

        convolve = weights @ img_kernels

        convolve = convolve.reshape(self.batchSize, ker_f_out, img_h, img_w)

        #add bias
        z_out = convolve.transpose() + np.tile(self.bias, (self.batchSize, 1)).T
        z_out = z_out.transpose()

        #add activation function
        a_out = self.reLU(z_out)
        self.z_out = z_out

        return z_out, a_out
    

    def backward_prop(self, delta, eta):
        """
        Perform backward propagation for the ConvLayer.

        Args:
            delta (ndarray): Delta value from the next layer.
            eta (float): Learning rate.

        Returns:
            ndarray: Delta value for the previous layer.
        """

        #delta computation (error)
        delta = delta * self.reLU_Derivative(self.z_out)
        
        #compute output dimensions
        out_h = self.img_h
        out_w = self.img_w
        pad = self.pad

        # sum over the error for each feature map as there is 
        # only one bias for each feature map

        #bias derivitive
        db = np.sum(delta, axis=(-2, -1))

        
        #weights derivitve
        img_kernels = self.pad_img[:, :, self.i, self.j]
        img_kernels = img_kernels.reshape((self.batchSize, self.ker_h * self.ker_w * self.ker_f_in, self.img_h * self.img_w))
        img_kernels = img_kernels.transpose((0, 2, 1))

        delta_reshape = delta.reshape(self.batchSize, self.ker_f_out, -1)

        dw = delta_reshape @ img_kernels
        dw = dw.reshape((self.batchSize, self.ker_f_out, self.ker_f_in, self.ker_h, self.ker_w))

        nabla_b = db.T.sum(axis=-1)
        nabla_w = dw.T.sum(axis=-1).T

        #cmputing error to backpropagate
        delta_padding = np.pad(delta, ((0, 0), (0, 0), (pad, pad), (pad, pad)))

        img_kernels = delta_padding[:, :, self.i, self.j]
        img_kernels = img_kernels.reshape((self.batchSize, self.ker_h * self.ker_w * self.ker_f_out, out_h * out_w))

        kernels = self.kernels.transpose((1, 0, 2, 3))
        kernels = np.flip(kernels, (len(self.kernels.shape) - 2, len(self.kernels.shape) - 1))
        weights = kernels.reshape((self.ker_f_in, self.ker_h * self.ker_w * self.ker_f_out))

        convolve = weights @ img_kernels
        convolve = convolve.reshape(self.batchSize, self.ker_f_in, out_h, out_w)

        delta_new = convolve


        #update weight and biases
        self.kernels -= (eta * (nabla_w / self.batchSize))
        self.bias -= (eta * (nabla_b / self.batchSize))

        return delta_new