# Convolutional Neural Network (CNN) from scratch.

## Motivation

The CNN framework was created as part of a University Project during my M.S.c. programm. The goal was to develop different Machine Learning methods from scratch to conduct experiments on its perfomance in part of a overall Project. As Neural Networks find themself as a subset of Machine Learning methods i choosed among others a CNN. The goal was to detect the verticies of a Sudoku on a grayscale image.

The Framework computations are only based on low level [Cupy](https://docs.cupy.dev/en/stable/install.html) (Numpy extension to enable GPU excelration) operations. If you want to use only Numpy you can easly do this by just replacing the cupy import with your numpy import.

> NOTE: The CNN framework was self-developed and should therefore be treated with caution with regard to the accuracy of the calculations. However, especially for the Sudoku detection and number recognition, calculation results were compared with the well-known pytorch framework and the calculations were confirmed. If you just need a simple, fast and highly customizable framework, use the [PyTorch](https://pytorch.org/get-started/locally/) framework.


## Install Cupy

Check the [cupy documentation](https://docs.cupy.dev/en/stable/install.html) to choose the correct cupy version compatible with your Cuda version. After you choosed a compatible version you can use pip for installing.

## Dataset

#### Sudoku Dataset

For the sudoku dataset pictures where take from a [Kaggel Dataset by Sudarshan s magaji](https://www.kaggle.com/datasets/macfooty/sudoku-box-detection/data).

Then 400 random images were annotated by labeling the corner vertices in a clockwise direction. For annotating [CVAT](https://www.cvat.ai/) tool was used.

#### MNIST Dataset

For trying to detect numbers the famous MNIST Dataset was used.

You can download it at [Kaggel](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) or at the [website](http://yann.lecun.com/exdb/mnist/) from the orignial creator.

Unpack all the files and put them in the Dataset folder.


## Training

While the Network was mainly created for detecting the verticies of a Soduku, it is build to support a variaty of architectures and with that a variaty of porpuses.

#### *Train a model:*

For training a model one can use the *train_mnist_numb_rec.py* as a basis. With this example you can understand how to init a Model and how to train.

1. Init Network

   ```
   N = Network(
   	#specify the loss function you want to use ("MSE", "CrossEntropy")
           loss= "CrossEntropy",
   	#specify your architecture by defining the Layer types (order is important)
           layers= [
           ConvLayer(inputSize=28, fM_n_in=1,fM_n_out=32,kernelSize=3, initMethod= "He"),
           MaxPoolingLayer(kernel_size=2),
           ConvLayer(inputSize=14, fM_n_in=32,fM_n_out=32,kernelSize=3, initMethod= "He"),
           MaxPoolingLayer(kernel_size=2),
           FlattenLayer(),
           FullyConnLayer(input_size=7*7*32,output_size=128, activation="reLU", initMethod="He"),
           DropoutLayer(0.2),
           FullyConnLayer(input_size=128,output_size=10, activation="softmax", initMethod="Xavier"),
        ]
   ```
2. Load Dataset

   To load a Dataset you can take a look at the load at the loadMNISTDataset method. If you have a custom Dataset you may have to implement a dataset loader yourself. Important is, that there are a **test** and a **train** set created at the end of the method and safed by self.val_dataset and self.train_dataset.

```
    N.loadMNISTDataset("./Dataset/train-images.idx3-ubyte", "./Dataset/train-labels.idx1-ubyte", "./Dataset/t10k-images.idx3-ubyte", "./Dataset/t10k-labels.idx1-ubyte",)
```

3. Train the Model

   To train the initialized network define the hyperperameters and then train the network.

```
    batch_size = 32
    epoch_nr = 20
    tr = 0.01
  
    N.train_network(batch_size,epoch_nr,tr, False)
```

4. Save model

   You can save the model by serializing the layer classes. The Network class provides a save and load method for this.

   > NOTE: To save the model, the simplest method was chosen due to time constraints. Therefore, the project uses the [Pickle](https://docs.python.org/3/library/pickle.html) library. This results in files with high memory consumption.
   

```
    filename = './CNN/saved_networks/SudokuDetect/'
    N.save_network(filename,epoch_nr,tr)
```
