from array import array
import struct
import time
from matplotlib import pyplot as plt
import cupy as np
import cv2 as cv
from pycocotools.coco import COCO
from ConvLayer import ConvLayer
from MaxPoolingLayer import MaxPoolingLayer
from FullyConnLayer import FullyConnLayer
from FlattenLayer import FlattenLayer
import pickle
import os
import copy 
import numpy as numpy


class Network:

    def __init__(self, layers, loss = "MSE"):
        if loss == "MSE":
            self.loss = self.MSECost
            self.loss_derivative = self.MSECostDerivative
        elif loss == "CrossEntropy":
            self.loss = self.CrossEntropyCost
            self.loss_derivative = self.CrossEntropyCostDerivative
        
        self.l = layers


    def loadMNISTDataset(self, images_filepath, labels_filepath, test_img_filepath, test_lab_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        labels = np.array(labels)

        new_labels = np.zeros((len(labels), 10))
        for i,label in enumerate(labels):
            # for label create new numpy with size 10 and set the label index to 1.0 all other to 0.0
            new_label = np.zeros((10))
            new_label[label] = 1.0
            new_labels[i] = new_label

        labels = new_labels
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())    

        images = []

        

        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28) / 255
            images[i][:] = np.array([img])

        images = np.array(images)
        
        

        self.train_dataset = list(zip(images, labels))
        labels = []
        with open(test_lab_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        labels = np.array(labels)
        new_labels = np.zeros((len(labels), 10))
        for i,label in enumerate(labels):
            # for label create new numpy with size 10 and set the label index to 1.0 all other to 0.0
            new_label = np.zeros((10))
            new_label[label] = 1.0
            new_labels[i] = new_label

        labels = new_labels
        with open(test_img_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())    

        images = []

        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)/ 255
            images[i][:] = np.array([img])

        images = np.array(images)


        self.val_dataset = list(zip(images, labels))

    

    def loadCocoDataset(self, coco_json_path, image_folder, normalize, max_pixel_value, imagewidthheight):
        coco = COCO(coco_json_path)
        # Initialize an empty list to store the transformed dataset
        self.max_pixel_vl = max_pixel_value
        self.imagewidthheight = imagewidthheight
        image_dataset = []
        label_dataset = []
        # Iterate through each annotation in the dataset
        for annotation in coco.dataset['annotations']:
            image_id = annotation['image_id']
            image_name = coco.loadImgs(image_id)[0]['file_name']
            segmentation = annotation['segmentation'][0]  # Assuming only one segmentation
            image = cv.imread(os.path.join(image_folder, image_name), cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (imagewidthheight,imagewidthheight))

            segmentation_values = np.array([round(float(coord)) for coord in segmentation])


            
            if normalize:
                # Normalize the image and label
                image = image / max_pixel_value
                segmentation = (segmentation_values * (imagewidthheight/300)) / imagewidthheight

            # Add the image and segmentation as a tuple to the dataset_list
            image_dataset.append([image])
            label_dataset.append(segmentation)
        
        image_dataset = np.array(image_dataset)
        label_dataset = np.array(label_dataset)

        dataset_list = list(zip(image_dataset, label_dataset))

        split_ratio = 0.8
        dataset_size = len(dataset_list)
        train_size = int(split_ratio * dataset_size)

        indices = list(range(dataset_size))
        numpy.random.seed(12)
        #cupy seed not working correctly
        #np.random.seed(12)
        numpy.random.shuffle(indices)

        train_idx, val_idx = indices[:train_size], indices[train_size:]

        self.val_dataset = [dataset_list[i] for i in val_idx]
        self.train_dataset = [dataset_list[i] for i in train_idx]

        rnd = np.random.randint(0, len(self.train_dataset), int(0.2*len(self.train_dataset)))
        for i in rnd:
            i = int(i)
            self.train_dataset[i] = (1 - self.train_dataset[i][0], self.train_dataset[i][1])
        
        # t = 1000 * time.time() # current time in milliseconds
        # np.random.seed(int(t) % 2**32)
        # print("Dataset loaded")

    def export_training_progress(self, train_cost_history, val_cost_history, eta, final_time):
        epochs = range(len(train_cost_history))
        plt.plot(epochs, train_cost_history, label='Training Cost per Vertex')
        plt.plot(epochs, val_cost_history, label='Validation Cost per Vertex')
        plt.xlabel('Epochs')
        plt.ylabel('Cost')
        plt.title('Training Progress')
        plt.legend()
        plt.xlim(0, len(train_cost_history))  # Set x-axis limits from 0 to length of the history
        plt.ylim(0, max(max(train_cost_history), max(val_cost_history)))  # Set y-axis limits from 0 to max cost

        strin = "Eta " + str(eta) + "_"
        for l in self.l:
            strin += l.getInfo()

        print(strin)
        plt.savefig(strin+ "_Time_"+ str(final_time)+ '.png' )  # Save the plot as an image file
    
    def train_network(self, mb_size, epochs, eta, vis_training):
        start = time.time()
        img, lab = zip(*self.train_dataset)
        td = np.array(img)
        td_org = td.copy()
        tl = np.array(lab)
        tl_org = tl.copy()

        img, lab = zip(*self.val_dataset)
        vd = np.array(img)
        vl = np.array(lab)
        n = len(td)

        tC = 10000
        vC = 10000
        same_cost_count = 0
        bestEpoch = self.l.copy()
        bestEpoch_idx = 0
        lowestCost = vC
        train_cost_history = np.zeros(epochs)
        val_cost_history = np.zeros(epochs)
        
        for e in range(epochs):
            rand = np.random.randint(0, n, n)
            #print(rand)
            td = td_org[rand]
            tl = tl_org[rand]
            c= 0


            for i in range(0, n, mb_size):

                c += self.update_mini_batch(td[i:i+mb_size], tl[i:i+mb_size],eta)

            tC_New = c/n
            print("Epoch {} training complete".format(e+1))

            #tC_New = self.totalCost(td, tl, mb_size)
            train_cost_history[e] = tC_New
            vC_New = self.totalCost(vd, vl, mb_size)
            val_cost_history[e] = vC_New

            if(tC_New == tC):
                same_cost_count += 1
                if(same_cost_count == 5):
                    print("Training complete because of no improvement")
                    bestEpoch_idx = e
                    self.l = bestEpoch
                    break
            else:
                same_cost_count = 0

            if(vC_New < lowestCost):
                bestEpoch_idx = e
                lowestCost = vC_New
                bestEpoch = copy.deepcopy(self.l)
            
            tC = tC_New
            vC = vC_New

            if(vis_training and (e % 300 == 0)):
                rdIdx = np.random.randint(0,n)
                rdm_train_exp = td[rdIdx]
                a = self.forward(np.array([rdm_train_exp]), False)
                image = self.grayscale_image(rdm_train_exp[0])
                a = self.scale_verticies(a)
                label = self.scale_verticies(tl[rdIdx])
                self.visualize_image_and_segmentation(image, a , label=label)
            
            print("(TrainData) Total cost per vertics is {:.5f}".format(tC))
            print("(ValData) Total cost per vertics is {:.5f}".format(vC))

        self.l = bestEpoch
        print('Network was set to best epoch() test_Cost')
        print("Best epoch was {:.5f}".format(bestEpoch_idx))
        print("Total cost per vertics is for test_set{:.5f}".format(lowestCost))
        print("Training complete")
        end = time.time()
        final_time = end - start
        self.export_training_progress(np.asnumpy(train_cost_history).tolist(), np.asnumpy(val_cost_history).tolist(), eta, final_time)
    
    def totalCost(self, td, tl, batch_s = 4):
        #forward for 4 images at a time
        c = 0
        for i in range(0, len(td), batch_s):
            a = self.forward(td[i:i+batch_s], False)
            c += self.cost(a, tl[i:i+batch_s])

        return c/len(td)


    
    def MSECost(self,a, y):
        return 1/a.shape[-1] * np.sum((a-y)**2)
    
    
    def MSECostDerivative(self,a, y):
        return 2 * (a-y) / a.shape[-1]
    
    
    #For multiclasses
    def CrossEntropyCost(self,a, y):
        return -np.sum(y * np.log(a))
    
    
    def CrossEntropyCostDerivative(self,a, y):
        return - (np.divide(y, a) - np.divide(1 - y, 1 - a))


    def cost(self, a, y):
        return self.loss(a, y)
    

    def cost_derivitiv(self, a, y):
        return self.loss_derivative(a, y)
    
    def predict(self, image, size, max_pixel_value):
        self.max_pixel_vl = max_pixel_value
        self.imagewidthheight = size
        if(len(image.shape)>2):
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = self.normalize_image(image)

        cv.imshow("img", np.asnumpy(image))
        cv.waitKey(0)
        cv.destroyAllWindows()
        runimage = image.reshape(1,1,image.shape[0],image.shape[1])

        a = self.forward(runimage, False).squeeze()
        
        return a
        # image = self.grayscale_image(image)
        # a = self.scale_verticies(a)


        # self.visualize_image_and_segmentation(image, a)

    def scale_verticies(self, a):
        return np.multiply(a, self.imagewidthheight)

    def normalize_verticies(self, a):
        return np.divide(a, self.imagewidthheight)

    def grayscale_image(self, img):
        return np.multiply(img, self.max_pixel_vl)
    
    def normalize_image(self, img):
        return np.divide(img, self.max_pixel_vl)

    def forward(self, a, training):
        z=a
        for l in self.l:
            z, a = l.forward_prop(z,a, training)
        return a
   

    def visualize_image_and_segmentation(self,image, segmentation, label = None):
        # Create a new figure
        plt.figure()
        image = np.asnumpy(image)
        segmentation  =np.asnumpy(segmentation).squeeze()
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)  
        x = segmentation[0::2]
        y = segmentation[1::2]
        
        x = np.asnumpy(x)
        y = np.asnumpy(y)
         # Define colors for each point
        colors = np.asnumpy(np.arange(len(x)))

        # Define labels for legend
        point_labels = ["TL", "TR", "BR", "RL"]

        # Plot markers for each point
        for i in range(len(x) - 1):
            plt.plot([x[i], x[i+1]], [y[i], y[i+1]], linestyle='-', color='red')  # Change color/style as needed

        for i in range(len(x)):
            plt.plot(x[i], y[i], marker='o', color=plt.cm.viridis(colors[i] / (len(x)-1)), label=point_labels[i])

        #show the label besides the image
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

        if(label is not None):
            label = np.asnumpy(label)
            x = label[0::2]
            y = label[1::2]
            
            x = np.asnumpy(x)
            y = np.asnumpy(y)

            # Plot the points with different colors and labels
            for i in range(len(x)):
                plt.plot(x[i], y[i], marker='o', linestyle='-', color='r', label=point_labels[i])


        # Show the figure
        plt.show()


    def evaluate_vertices(self, batch_size = 2):
        img, lab = zip(*self.val_dataset)

        vd = np.array(img)
        vl = np.array(lab)
        total_a = []
        for i in range(0, len(vd), batch_size):
            a = self.forward(vd[i:i+batch_size], False)
            for j in range(len(a)):
                total_a.append(a[j])
        
        total_a = np.array(total_a)
        total_a = self.scale_verticies(total_a)
        total_a = np.asnumpy(total_a)
        vl = self.scale_verticies(vl)
        vl = np.asnumpy(vl)

        mean_diff = 0
        for i in range(0,len(total_a),1):
            difference = 0
            for j in range(0, len(total_a[i]),2):
                point_a = (total_a[i][j], total_a[i][j+1])
                point_b = (vl[i][j], vl[i][j+1])
                difference += numpy.linalg.norm(numpy.array(point_a) - numpy.array(point_b))
            mean_diff += difference/4

        mean_diff = mean_diff/len(total_a)
        print("Mean difference between predicted and actual verticies: ", mean_diff)
        
        for j in range(len(total_a)):
            img = self.grayscale_image(vd[j][0])
            self.visualize_image_and_segmentation(img, total_a[j], None)

    

    def evaluate_classification(self, batch_size = 32):
        img, lab = zip(*self.val_dataset)

        vd = np.array(img)
        vl = np.array(lab)

        #forward in batch size
        true_counter = 0
        
        res = np.zeros((len(vd), 10))        
        for i in range(0, len(vd), batch_size):
            a = self.forward(vd[i:i+batch_size], False)
            res[i:i+batch_size] = a
        
        res_np = np.asnumpy(res)

        for i in range(len(res_np)):
            if(numpy.argmax(res_np[i]) == numpy.argmax(vl[i])):
                true_counter += 1 

        print("Accuracy: ", true_counter/len(vd))
        print("True: ", true_counter)
        print("False: ", len(vd) - true_counter)



    def update_mini_batch(self, images, lab, eta):
        y = lab
        a = self.forward(images, True)
        cost = self.cost(a,y)

        #BACKPROPAGATION
        delta = self.cost_derivitiv(a,y)
        #update w and b
        for l in reversed(self.l):
            delta = l.backward_prop(delta, eta)

        return cost

    def save_network(self, file_name, e_n, tr):
        img, lab = zip(*self.val_dataset)
        vd = np.array(img)
        vl = np.array(lab)
        tC = self.totalCost(vd, vl)
        file_name = file_name + "ToEP{}".format(e_n) + "TR{}".format(tr) + self.getLayersInfo() + "{:.5f}".format(tC) + ".pkl"
        with open(file_name, 'wb') as outp:
            pickle.dump(self.l, outp, pickle.HIGHEST_PROTOCOL)


    def getLayersInfo(self):
        strin = ""
        for l in self.l:
            strin += l.getInfo()
        return strin

    
    def load_network(file_name):
        with open(file_name, 'rb') as inp:
            layer = pickle.load(inp)

        return Network(layer)