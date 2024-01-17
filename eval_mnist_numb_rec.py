from CNN.Network import Network

import cv2 as cv

model = "./MachineLearning/CNN/saved_networks/NumberDetect/NumberDetectToEP20TR0.01C32MPC32MPFlatFC128DL0.4FC100.02841.pkl"

N = Network.load_network(model)
N.loadMNISTDataset("./Dataset/train-images.idx3-ubyte", "./Dataset/train-labels.idx1-ubyte", "./Dataset/t10k-images.idx3-ubyte", "./Dataset/t10k-labels.idx1-ubyte",)

N.evaluate_classification(32)
