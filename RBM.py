# Importing the libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# Importing the dataset
movies = pd.read_csv('ml-latest/ml-latest/movies.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
users = pd.read_csv('ml-latest/ml-latest/tags.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')
ratings = pd.read_csv('ml-latest/ml-latest/ratings.csv', sep = ',', header = None, engine = 'python', encoding = 'latin-1')

# Preparing the training set and the test set
training_set_RBM = pd.read_csv('ml-100k/ml-100k/u2.base', delimiter = '\t')
training_set_RBM  = np.array(training_set_RBM , dtype = 'int')
test_set_RBM  = pd.read_csv('ml-100k/ml-100k/u2.test', delimiter = '\t')
test_set_RBM  = np.array(test_set_RBM , dtype = 'int')

# Getting the number of users and movies
total_number_of_users = int(max(max(training_set_RBM[:,0]), max(test_set_RBM[:,0])))
total_number_of_movie = int(max(max(training_set_RBM[:,1]), max(test_set_RBM[:,1])))

#we create the data into an array with lines in users and colomns in movies
def convert(data_100K):
    new_data_100K = []
    for RBM_users_100K in range(1, total_number_of_users+ 1):
        RBM_movies_100K = data_100K[:,1][data_100K[:,0] == RBM_users_100K]
        RBM_ratings_100K = data_100K[:,2][data_100K[:,0] == RBM_users_100K]
        ratings = np.zeros(total_number_of_movie)
        ratings[RBM_movies_100K - 1] = RBM_ratings_100K
        new_data_100K.append(list(ratings))
    return new_data_100K
training_set_RBM = convert(training_set_RBM)
test_set_RBM = convert(test_set_RBM)

# Converting the data into Torch tensors
training_set_RBM= torch.FloatTensor(training_set_RBM)
test_set_RBM  = torch.FloatTensor(test_set_RBM )

# Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked)
training_set_RBM[training_set_RBM == 0] = -1
training_set_RBM[training_set_RBM == 1] = 0
training_set_RBM[training_set_RBM == 2] = 0
training_set_RBM[training_set_RBM >= 3] = 1
test_set_RBM [test_set_RBM  == 0] = -1
test_set_RBM [test_set_RBM  == 1] = 0
test_set_RBM [test_set_RBM  == 2] = 0
test_set_RBM [test_set_RBM  >= 3] = 1

#Creating the rmb neural network
class RestrictedBoltzmannMachine():
    def __init__(self, numvis, numhid):
        self.W = torch.randn(numhid, numvis)#the prob of visible node given hidden nodes
        self.a = torch.randn(1, numhid)#the bias of the probablity of the hidden nodes given the visible. Note: 2d bias bcos of tensor
        self.b = torch.randn(1, numvis)#the bias of the probablity of the visible nodes given the hidden, 2d tensor with 1 rep the bach 
    def samples_hidden_gibs(self,x): #this samples the prob of hidden node given the visible nodes, using gibs sampling 
        wx = torch.mm(x, self.W.t())#the prob of hidden node given the visible nodes
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def samples_visible_gibs(self,y): #this samples the prob of visble node given the hidden nodes
        wy = torch.mm(y, self.W)#defining the variable
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, vis0, visk, phidd0, phiddk): #contractive divergence
        self.W += (torch.mm(vis0.t(), phidd0) - torch.mm(visk.t(), phiddk)).t()
        self.b += torch.sum((vis0 - visk), 0)
        self.a += torch.sum((phidd0 - phiddk), 0)

numvis = len(training_set_RBM[0])
numhid = 100 #number of hidden node is the number of features
batch_size = 100
rbm= RestrictedBoltzmannMachine(numvis, numhid)#remember the sequential class

#Evakuation of RBM model using RMSE(Root Mean Squared Error)   
# Training the RBM
number_of_epoch = 20
for epoch in range(1, number_of_epoch + 1):
    train_loss = 0
    s = 0.
    for RBM_users_100K in range(0, total_number_of_users - batch_size, batch_size):
        visk = training_set_RBM[RBM_users_100K:RBM_users_100K+batch_size]
        vis0 = training_set_RBM[RBM_users_100K:RBM_users_100K+batch_size]
        phidd0,_ = rbm.samples_hidden_gibs(vis0)
        for k in range(20):
            _,hidd_k_step = rbm.samples_hidden_gibs(visk)#HIDDEN NODE AT KTH STEP
            _,visk = rbm.samples_visible_gibs(hidd_k_step)
            visk[vis0<0] = vis0[vis0<0]
        phiddk,_ = rbm.samples_hidden_gibs(visk)
        rbm.train(vis0, visk, phidd0, phiddk)
        train_loss += torch.mean(torch.abs(vis0[vis0>=0] - visk[vis0>=0]**2))
        s += 1.
    print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    
# Testing the RBM
test_loss = 0.0
s = 0.0
for RBM_users_100K in range(total_number_of_users):
    v = training_set_RBM[RBM_users_100K:RBM_users_100K+1]
    vt = test_set_RBM [RBM_users_100K:RBM_users_100K+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.samples_hidden_gibs(v)
        _,v = rbm.samples_visible_gibs(h)
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))


