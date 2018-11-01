import pandas as pd
import numpy as np
# import pickle
# import os
from keras.utils import Sequence
class BalancedDataGenerator_binary(Sequence):
    """ A class to generate balanced training dataset on demand"""
    def __init__(self,features,labels,scalerLabel,batch_size,shuffle=True,classProbability=[1,1]):
        """
        Initialize class
        :param features:features
        :param labels: labels
        :param scalerLabel: a 1d array with scalers, 0 for negative samples and 1 for positive samples
        :param batch_size: batch size
        :param shuffle: whether to shuffle data for each epoch
        :param classProbability: [false class pobability, true probability]
        """
        self.x = features
        self.y = labels
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.nsample=len(features)
        self.classProb=classProbability
        self.ind_pos_augment = None
        # convert from onehot to class label, one per sample
        # maxClass=np.max(np.argmax(labels,axis=2),axis=1)
        maxClass=scalerLabel

        # separate positive and negative samples
        self.ind_pos= np.where(maxClass==1)[0]
        self.ind_neg= np.where(maxClass==0)[0]
        self.n_neg = int(len(self.ind_neg))
        self.n_pos = int(len(self.ind_pos))
        self.n_pos_augment = int(self.n_neg*classProbability[1]/classProbability[0])

        # calculate numer of samples of each class in a balanced batch
        self.n_neg_batch = int(batch_size*classProbability[0]/np.sum(self.classProb))
        self.n_pos_batch = int(batch_size*classProbability[1]/np.sum(self.classProb))
        self.on_epoch_end()

        self.index_batch=None
    def __len__(self):
        # number of batch per epoch: go over all
        return int(np.floor(self.n_neg / float(self.n_neg_batch)))
        # return int(np.floor(len(self.y) / float(self.batch_size)))


    def on_epoch_end(self):
        'Shuffling index after each epoch'
        # self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.ind_pos)
            np.random.shuffle(self.ind_neg)
        self.ind_pos_augment=self.pad_augment_index()

    def pad_augment_index(self):
    # shuffle posiive samples to match the size of negative samples
        ind_pos_augment = self.ind_pos
        # nsample_pad = self.n_pos

        # print('size of the true index is')
        # print(len(ind_pos_augment))
        while len(ind_pos_augment) < self.n_pos_augment:
            # print(len(ind_pos_augment))
            # append a shuffled index array to the end until the length of positive index > length of negative index
            ind_pos_augment = np.concatenate((ind_pos_augment,np.random.permutation(self.ind_pos)))
        return ind_pos_augment

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_pos = self.ind_pos_augment[index * self.n_pos_batch:(index + 1) * self.n_pos_batch]
        index_neg = self.ind_neg[index * self.n_neg_batch:(index + 1) * self.n_neg_batch]
        self.index_batch = np.concatenate((index_pos, index_neg))
        x = self.x[self.index_batch]
        y = self.y[self.index_batch]
        return x, y


class BalancedDataGenerator_multifeature(Sequence):
    """ A class to generate balanced training dataset on demand, for multiple input feature types"""
    def __init__(self,features,labels,scalerLabel,batch_size,shuffle=True,classProbability=[1,1]):
        """
        Initialize class
        :param features:list of features
        :param labels: labels
        :param scalerLabel: a 1d array with scalers, 0 for negative samples and 1 for positive samples
        :param batch_size: batch size
        :param shuffle: whether to shuffle data for each epoch
        :param classProbability: [false class pobability, true probability]
        """
        self.x = features
        self.nfeature = len(self.x)
        self.y = labels
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.nsample=len(features)
        self.classProb=classProbability
        self.ind_pos_augment = None
        # convert from onehot to class label, one per sample
        # maxClass=np.max(np.argmax(labels,axis=2),axis=1)
        maxClass=scalerLabel

        # separate positive and negative samples
        self.ind_pos= np.where(maxClass==1)[0]
        self.ind_neg= np.where(maxClass==0)[0]
        self.n_neg = int(len(self.ind_neg))
        self.n_pos = int(len(self.ind_pos))
        self.n_pos_augment = int(self.n_neg*classProbability[1]/classProbability[0])

        # calculate numer of samples of each class in a balanced batch
        self.n_neg_batch = int(batch_size*classProbability[0]/np.sum(self.classProb))
        self.n_pos_batch = int(batch_size*classProbability[1]/np.sum(self.classProb))
        self.on_epoch_end()
        self.index_batch=None
    def __len__(self):
        # number of batch per epoch: go over all
        return int(np.floor(self.n_neg / float(self.n_neg_batch)))
        # return int(np.floor(len(self.y) / float(self.batch_size)))


    def on_epoch_end(self):
        'Shuffling index after each epoch'
        # self.indexes = np.arange(self.nsamples)
        if self.shuffle == True:
            np.random.shuffle(self.ind_pos)
            np.random.shuffle(self.ind_neg)
        self.ind_pos_augment=self.pad_augment_index()

    def pad_augment_index(self):
    # shuffle posiive samples to match the size of negative samples
        ind_pos_augment = self.ind_pos
        # nsample_pad = self.n_pos

        # print('size of the true index is')
        # print(len(ind_pos_augment))
        while len(ind_pos_augment) < self.n_pos_augment:
            # print(len(ind_pos_augment))
            # append a shuffled index array to the end until the length of positive index > length of negative index
            ind_pos_augment = np.concatenate((ind_pos_augment,np.random.permutation(self.ind_pos)))
        return ind_pos_augment

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_pos = self.ind_pos_augment[index * self.n_pos_batch:(index + 1) * self.n_pos_batch]
        index_neg = self.ind_neg[index * self.n_neg_batch:(index + 1) * self.n_neg_batch]
        self.index_batch = np.concatenate((index_pos, index_neg))
        # index_batch
        # index_batch = self.getIndex(index)
        # print('batch index are')
        # print(self.index_batch)
        batchFeatures=[]
        # print(index_neg)
        for i in range(self.nfeature):
            batchFeatures.append(self.x[i][self.index_batch])
        batchLabels = self.y[self.index_batch]
        # print('y is')
        # print(batchLabels)
        return batchFeatures, batchLabels

def main():

    from dataGenerator import BalancedDataGenerator_binary
    npos = 100
    nneg = 1000
    feature_neg = np.random.rand(nneg, 30, 3)
    feature_pos = np.random.rand(npos, 30, 3)
    label_neg = np.zeros((nneg, 30, 2))
    label_pos = np.random.rand(npos, 30, 2)

    # label_neg = np.zeros((nneg,))
    # label_pos = np.ones((npos,))
    features = np.concatenate((feature_neg,feature_pos),axis=0)
    labels = np.concatenate((label_neg,label_pos),axis=0)
    maxClass = np.max(np.argmax(labels, axis=2), axis=1)


    gen = BalancedDataGenerator_binary(features,labels,scalerLabel=maxClass,
                                       batch_size=32,shuffle=True,classProbability=[2,1])

    nbatch = gen.__len__()
    for i in range(nbatch):
        xb,yb = gen.__getitem__(i)
        maxClassBatch  =np.max(np.argmax(yb, axis=2), axis=1)
        print('percentage of positive sample is '+ str(sum(maxClassBatch)/len(yb)))


if __name__ =='__main__':
    main()
