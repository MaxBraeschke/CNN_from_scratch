from CNN.ConvLayer import ConvLayer
from CNN.MaxPoolingLayer import MaxPoolingLayer
from CNN.FullyConnLayer import FullyConnLayer
from CNN.FlattenLayer import FlattenLayer
from CNN.DropoutLayer import DropoutLayer
from CNN.Network import Network

  
def main():
    """
    This is the main function that executes the training process of a convolutional neural network (CNN).
    It initializes the network architecture, loads the dataset, trains the network, and saves the trained network.
    It is important that the initialization of the layers has to be matched to each other.
    """
    N = Network(
        loss= "CrossEntropy",
        layers= [
        ConvLayer(inputSize=28, fM_n_in=1,fM_n_out=32,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        ConvLayer(inputSize=14, fM_n_in=32,fM_n_out=32,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        FlattenLayer(),
        FullyConnLayer(input_size=7*7*32,output_size=128, activation="reLU", initMethod="He"),
        DropoutLayer(0.2),
        FullyConnLayer(input_size=128,output_size=10, activation="softmax", initMethod="Xavier"), 
     ])
    
    N.loadMNISTDataset("./Dataset/train-images.idx3-ubyte", "./Dataset/train-labels.idx1-ubyte", "./Dataset/t10k-images.idx3-ubyte", "./Dataset/t10k-labels.idx1-ubyte",)
    batch_size = 32
    epoch_nr = 20
    tr = 0.01
    
    N.train_network(batch_size,epoch_nr,tr, False)
    
    filename = './CNN/saved_networks/SudokuDetect/'
    N.save_network(filename,epoch_nr,tr)


main()

