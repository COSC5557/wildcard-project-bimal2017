import numpy as np
from scipy.io import loadmat
import torch
import tkinter
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
import os
import time
from data_loading import loading, separate_dataset, freq_data
from model import Net, Simple1DCNN
from utils import accuracy, testing
from torch.utils.data import DataLoader


if __name__ == '__main__':
    
    torch.manual_seed(0);  np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # parameters are defined below
    network = 'CNN'
    epoch =200   # number of epochs 
    mini_batch = 30         # number of mini-batches
    learning_rate = 1e-3    # SGD learning rate 
    momentum = 0.5          # SGD momentum term 
    n_hidden1 = 25          # number of hidden units in first hidden layer
    n_hidden2 = 25          # number of hidden units in second hidden layer 
    n_output = 1            # number of output units
    frac_train = 0.80       # fraction of data to be used as training set
    dropout_rate = 0.2      # dropout rate 
    weight_initializer = 0.05   # weight initializer
    dropout_decision = True   
    w_lambda = 0.0005           # weight decay parameter
    tolerance = 0.1             # tolerance 
    

    save_model = False

    data_path= data_path = r"C:\Users\UW-User\Downloads\Projects\\"

    # loading the data
    dataset = freq_data(data_path)  # loads data from the freq_data class
    #dataset.info
    print('the length of the dataset = ', len(dataset))

    train_num = int(frac_train * len(dataset))  # number of data for training
    test_num = len(dataset) - train_num  # number of data for validating
    max_batches = epoch * int(train_num / mini_batch)

     
     

    # splitting into training and validation dataset
    training, validation = torch.utils.data.random_split(dataset, (train_num, test_num))

    # load separate training and validating dataset 
    train_loader = DataLoader(training, batch_size=mini_batch, shuffle=True)
    validation_loader = DataLoader(validation, batch_size=mini_batch, shuffle=False)

    
    n_inp = len(training[0][0])
    n_hid1 = n_hidden1
    n_hid2 = n_hidden2
    n_out = n_output


    if (network == 'CNN'):
        net = Simple1DCNN().double().to(device)
    else:
        net = Net(n_inp, n_hid1, n_hid2, n_out, dropout_rate, weight_initializer, dropout_decision).to(device)

    print(net)
    class Simple1DCNN(torch.nn.Module):
        def __init__(self):
            super(Simple1DCNN, self).__init__()
            self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=10, kernel_size=3)
            self.act = torch.nn.ReLU()
            self.layer2 = torch.nn.Conv1d(in_channels=10, out_channels=20, kernel_size=3)
            self.fc1 = torch.nn.Linear(20 * 398, 800)
            self.fc2 = torch.nn.Linear(800, 50)
            self.fc3 = torch.nn.Linear(50, 1)
            self.conv2_drop = torch.nn.Dropout(0.5)

        

        def forward(self, x):
            x = self.act(self.layer1(x))
            x = self.act(self.conv2_drop(self.layer2(x)))
            x = x.view(-1, x.shape[1] * x.shape[-1])
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            x = self.fc3(x)
            return x


    net = net.train()  # set the network to training mode
    criterion = torch.nn.MSELoss()  # set the loss criterion
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=w_lambda)
    print("Training is starting!!!!!!!!! \n")
    print("h{}_lr{}_lam{}_bat_{}".format(n_hidden1, learning_rate, w_lambda, mini_batch))


    weight_ih = []  # It stores the weights from input to hidden layer
    weight_ho = []  # It stores the weights from hidden unit to output unit
    train_losses = []  #   training losses are stored here
    val_losses = []  # validation losses are stored here
    test_losses = []  # storing the validation losses
    min_val_RMSE = 1e5  # finding min validation RMSE
    min_R_epoch = 1e5  # initializing to find the epoch with min validation RMSE
    counter = []  # it stores the different epochs that gives validation accuracy
    t_correct = []
    t_acc = []
    v_acc = []

    t0 = time.time()
    for ep in range(epoch):
        train_loss = []  # saves the batch training loss for each epoch
        t_item = 0
        t_correct = []
        for batch_idx, (data, target) in enumerate(train_loader):
            t_item += len(target)
            data, target = data.to(device), target.to(device)  
            if (network == 'CNN'):
                X = data.double().unsqueeze(1)  
                Y = target.double().view(-1, 1)  # converting into 2-d tensor
            else:
                X = data.float()  # pytorch input should be float -> awkward right?
                Y = target.float().view(-1, 1)  # converting into 2-d tensor
            optimizer.zero_grad()  # making the gradient zero before optimizing
            oupt = net(X)  # neural network output
            loss_obj = criterion(oupt, Y)  # loss calculation
            loss_obj.backward()  # back propagation
            optimizer.step()  # remember w(t) = w(t-1) - alpha*cost 

            train_loss.append(loss_obj.item())  # batch losses 
            correct = torch.sum((torch.abs(oupt - Y) < torch.abs(0.1 * Y)))
            t_correct.append(correct)

        if (network == 'CNN'):
            weight_ho.append((net.fc3.weight.data.clone().cpu().numpy()))
        else:
            weight_ho.append(np.reshape(net.oupt.weight.data.clone().cpu().numpy(), (1, n_hid2 * n_out)))

        t_result = ((sum(t_correct)).item() / t_item)
        t_acc.append(t_result)

        # getting the training loss for each epoch
        train_loss_avg = sum(train_loss) / len(train_loss)  # batch averaging
        train_losses.append([train_loss_avg])  # saving average batch loss for each epoch

    
        net = net.eval()  # set the network to evaluation mode
        val_acc, val_RMSE, vali_loss = accuracy(net, validation_loader, tolerance, criterion, device, network, eval=False)
        val_losses.append([vali_loss.item()])  # validation loss on entire samples for each epoch
        v_acc.append(val_acc.item() / 100)

        # find the epoch that gives minimum validation loss
        if val_RMSE < min_val_RMSE:
            min_val_RMSE = val_RMSE
            min_R_epoch = ep

        # set the network to training mode after validation
        net = net.train()
        if (save_model) and val_RMSE<=0.25:
            counter.append((ep))

        print("epoch = %d" % ep, end="")
        print("  train loss = %7.4f" % train_loss_avg, end="")
        print("  val_accuracy = %0.2f%%" % val_acc, end="")
        print("  val_RMSE = %7.4f" % val_RMSE, end="")
        print("  val_loss = %7.4f" % vali_loss.item())  


    
    print("Training complete \n")
    print("Time taken = {}".format(time.time() - t0))
    print(" min RMSE = {} at {} epoch \n".format(min_val_RMSE, min_R_epoch))

    ########Evaluating the model 

    net = net.eval()  # set eval mode
    acc_val, val_RMSE, _ = accuracy(net, validation_loader, tolerance, criterion, device, network, eval=True)
    print('validation accuracy with {} tolerance = {:.2f} and RMSE = {:.6f}\n'
          .format(tolerance, acc_val, val_RMSE))

    
    #Plotting the results
    title_font = {'fontname': 'Arial', 'size': '16', 'color': 'black', 'weight': 'normal',
                  'verticalalignment': 'bottom'}
    axis_font = {'fontname': 'Arial', 'size': '16'}

    label_graph = ['train_loss', 'val_loss', 'fitted_train_loss', 'fitted_val_loss', 'test_loss']
    losses = np.squeeze(train_losses)
    val_losses = np.squeeze(val_losses)

    t_x = np.arange(len(losses))
    v_x = np.arange(len(val_losses))

    plt.figure()
    plt.plot(t_x, losses, label=label_graph[0], c='yellow', linewidth='4')
    plt.plot(v_x, val_losses, label=label_graph[1], c='red', linewidth='1')
    plt.axvline(x=min_R_epoch, color='r', linestyle='--', linewidth=3)
    plt.ylabel("Mean Squared Error", **axis_font)
    plt.xlabel("Number of epochs", **axis_font)
    plt.xlim(0, len(t_x))
    plt.grid(linestyle='-', linewidth=0.5)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.legend()
    plt.show()
   
    plt.figure()
    plt.plot(t_x, t_acc, label='training accuracy', c='blue', linewidth='4')
    plt.plot(v_x, v_acc, label='testing accuracy', c='red', linewidth='1')
    plt.ylabel("Accuracy of data", **axis_font)
    plt.xlabel("Number of epochs", **axis_font)
    plt.xlim(0, len(t_x))
    plt.grid(linestyle='-', linewidth=0.4)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.rcParams['agg.path.chunksize'] = 1000
    plt.legend()
    plt.show()
    
    plt.figure()
    weight_ho = np.reshape(weight_ho, (np.shape(weight_ho)[0], np.shape(weight_ho)[2]))
    weights_ho_num = int(np.shape(weight_ho)[1])
    for i in range(0, weights_ho_num):
        plt.plot(weight_ho[:, i], color='magenta')
    plt.grid(linestyle='-', linewidth=0.4)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel("weights from hidden to output layer", **axis_font)
    plt.xlabel("no of epochs", **axis_font)
    plt.xlim(0, epoch)
    plt.rcParams['agg.path.chunksize'] = 10000
    plt.show()

    #######################  REFERENCES ################################
    #https://cs.colby.edu/courses/F22/cs343/projects/p3cnn/p3cnn.html
    #https://cnvrg.io/cnn-tensorflow/
    #https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_convolutional_neuralnetwork/
    #https://www.youtube.com/watch?v=WvoLTXIjBYU
    