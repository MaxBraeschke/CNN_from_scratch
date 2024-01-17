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
        loss= "MSE",
        
        layers= [
        ConvLayer(inputSize=256, fM_n_in=1,fM_n_out=16,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        ConvLayer(inputSize=128, fM_n_in=16,fM_n_out=16,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        ConvLayer(inputSize=64, fM_n_in=16,fM_n_out=32,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        ConvLayer(inputSize=32, fM_n_in=32,fM_n_out=64,kernelSize=3, initMethod= "He"),
        MaxPoolingLayer(kernel_size=2),
        DropoutLayer(0.2),
        FlattenLayer(),
        FullyConnLayer(input_size=16*16*64,output_size=64, activation="reLU", initMethod="He"), #preset_w = fc3_w, preset_b = fc3_b),
        DropoutLayer(0.4),
        FullyConnLayer(input_size=64,output_size=8, activation="linear", initMethod="Xavier"), #preset_w = fc3_w, preset_b = fc3_b),
     ])
    
    N.loadCocoDataset("./Dataset/400_dataset.json", "./Dataset/400random", True, 255, 256)
    batch_size = 2
    epoch_nr = 400
    tr = 0.01       
    
    N.train_network(batch_size,epoch_nr,tr, False)
    
    filename = './MachineLearning/CNN/saved_networks/SudokuDetect/'
    N.save_network(filename,epoch_nr,tr)


main()

