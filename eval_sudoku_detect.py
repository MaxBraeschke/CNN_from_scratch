from CNN.Network import Network

import cv2 as cv

model = "./MachineLearning/CNN/saved_networks/SudokuDetect/ToEP400TR0.01C16MPC16MPC32MPC64MPDL0.2FlatFC64DL0.4FC80.01423.pkl"

N = Network.load_network(model)
N.loadCocoDataset("./Dataset/400_dataset.json", "./Dataset/400random", True, 255, 256)

N.evaluate_vertices(2)