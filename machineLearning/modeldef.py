from keras.models import Sequential,Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.core import Reshape, Permute
from keras.layers import Merge,concatenate
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
# from keras.optimizers import SGD
import numpy as np


def model_cnn_lstm_adam_binary(inputShape,batchSize=None,stateful=False):
    """stateful training option"""
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    # numClasses=1

    model = Sequential()

    model.add(Convolution2D(4, (2, 5), padding='same', batch_input_shape=(batchSize,)+inputShape))
    #model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    convOutShape = model.layers[-1].output_shape
    print(convOutShape)
#     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    model.add(Reshape((convOutShape[1],np.prod(convOutShape[2:4]))))
#     model.add(Permute((2, 1)))
    model.add(LSTM(48, return_sequences=True, stateful=stateful))
    #model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('sigmoid'))
#     model.add(Activation('softmax'))
    return model, optimizer, loss


def model_cnn_lstm_adam_binary_dropout(inputShape,batchSize=None,stateful=False,dropout=0):
    """stateful training option"""
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    # numClasses=1

    model = Sequential()

    model.add(Convolution2D(4, (2, 5), padding='same', batch_input_shape=(batchSize,)+inputShape))
    #model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    convOutShape = model.layers[-1].output_shape
    print(convOutShape)
#     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    model.add(Reshape((convOutShape[1],np.prod(convOutShape[2:4]))))
    model.add(LSTM(48, return_sequences=True, stateful=stateful))
    model.add(Dropout(dropout))
    model.add(LSTM(48, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('sigmoid'))
#     model.add(Activation('softmax'))
    return model, optimizer, loss


def model_cnn_lstm_adam(inputShape, numClasses=1, batchSize=None,stateful=False):
    """stateful training option"""
    optimizer = 'adam'
    loss = 'binary_crossentropy'

    model = Sequential()

    model.add(Convolution2D(4, (2, 5), padding='same', batch_input_shape=(batchSize,)+inputShape))
    #model.add(MaxPooling2D(pool_size=(1, 3)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.25))
    convOutShape = model.layers[-1].output_shape
    print(convOutShape)
#     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    model.add(Reshape((convOutShape[1],np.prod(convOutShape[2:4]))))
#     model.add(Permute((2, 1)))
    model.add(LSTM(48, return_sequences=True, stateful=stateful))
    #model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(numClasses)))
    # model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    return model, optimizer, loss


# ------------------------mixed data model------------------------------

def model_branched_cnn_mixed_lstm_binary(input1Shape, input2Shape, outputShape, numFilter=4, batchSize=None,
                                         stateful=False,
                                         dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    branch1.add(Reshape((ntOut, -1)))
    # branch1.add(Dropout(0.2))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    # branch2.add(Dropout(0.2))
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    # branch2.add(Dropout(0.2))

    model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    model.add(Merge([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1.output,branch2.output],axis=2))

    # keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    model.add(LSTM(48, return_sequences=True, stateful=stateful))
    # model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    model.add(Activation('sigmoid'))

    return model, optimizer, loss


def model_branched_cnn_mixed_lstm_regression(input1Shape, input2Shape, outputShape,
                                             numFilter=4, batchSize=None,stateful=False,
                                            dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'mean_squared_error'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    branch1.add(Reshape((ntOut, -1)))
    # branch1.add(Dropout(0.2))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    # branch2.add(Dropout(0.2))
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    # branch2.add(Dropout(0.2))

    model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    model.add(Merge([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1.output,branch2.output],axis=2))

    # keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    model.add(LSTM(48, return_sequences=True, stateful=stateful))
    # model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=True))
    # model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))

    return model, optimizer, loss

from keras.layers import Input


def model_branched_cnn_mixed_lstm_binary_functional(input1Shape, input2Shape, outputShape, numFilter=4, batchSize=None,
                                                    stateful=False,
                                                    dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]
    input1 = Input(batch_shape=(batchSize,) + input1Shape)
    input2 = Input(batch_shape=(batchSize,) + input2Shape)

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    # branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    branch1.add(Reshape((ntOut, -1)))
    # branch1.add(Dropout(0.2))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    # branch2.add(Dropout(0.2))
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    # branch2.add(Dropout(0.2))
    output1 = branch1(input1)
    output2 = branch2(input2)

    #     model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    #     model.add(keras.layers.merge.concatenate([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1,branch2],axis=2))
    mergedInput = concatenate([output1, output2], axis=2)

    #     keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    #     model.add(LSTM(48, return_sequences=True, stateful=True))
    X = LSTM(48, return_sequences=True, stateful=stateful)(mergedInput)
    # model.add(Dropout(0.2))
    #     model.add(LSTM(48, return_sequences=True))
    X = LSTM(48, return_sequences=True)(X)
    # model.add(Dropout(0.2))
    #     model.add(TimeDistributed(Dense(1)))
    X = TimeDistributed(Dense(1))(X)
    #     model.add(Activation('sigmoid'))
    output = Activation('sigmoid')(X)

    model = Model(inputs=[input1, input2], outputs=output)

    return model, optimizer, loss


def model_branched_cnn_mixed_lstm_regression_padding(input1Shape, input2Shape, outputShape, numFilter=4, numUnitLSTM=16,
                                                     batchSize=None,
                                                     stateful=False,
                                                     dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'mean_squared_error'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]
    input1 = Input(batch_shape=(batchSize,) + input1Shape)
    input2 = Input(batch_shape=(batchSize,) + input2Shape)

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))

    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    convOutShape1 = branch1.layers[-1].output_shape
    branch1.add(Reshape((convOutShape1[1], np.prod(convOutShape1[2:4]))))
    #     integer multiple of output shape

    nPadTo = int(np.ceil(convOutShape1[1] / ntOut) * ntOut)
    nPadding = (nPadTo - convOutShape1[1])
    branch1.add(ZeroPadding1D(padding=(0, nPadding)))
    print('original size: ')
    print(convOutShape1[1])
    print('padding to multiples of:')
    print(ntOut)

    print('new shape: ')
    print(branch1.layers[-1].output_shape)
    branch1.add(Reshape((ntOut, -1)))
    branch1.add(Dropout(dropout))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    branch2.add(MaxPooling2D(pool_size=(1, 2)))
    branch2.add(Activation('relu'))
    branch2.add(Dropout(dropout))
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    branch2.add(MaxPooling2D(pool_size=(1, 2)))
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    branch2.add(Dropout(dropout))
    output1 = branch1(input1)
    output2 = branch2(input2)

    #     model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    #     model.add(keras.layers.merge.concatenate([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1,branch2],axis=2))
    mergedInput = concatenate([output1, output2], axis=2)

    #     keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    #     model.add(LSTM(48, return_sequences=True, stateful=True))
    X = LSTM(numUnitLSTM, return_sequences=True, stateful=stateful)(mergedInput)
    X = Dropout(dropout)(X)
    #     model.add(LSTM(48, return_sequences=True))
    X = LSTM(numUnitLSTM, return_sequences=True)(X)
    X = Dropout(dropout)(X)

    #     model.add(TimeDistributed(Dense(1)))
    output = TimeDistributed(Dense(1))(X)
    #     model.add(Activation('sigmoid'))

    model = Model(inputs=[input1, input2], outputs=output)

    return model, optimizer, loss


from keras.layers import ZeroPadding1D
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.core import Reshape, Permute
from keras.layers import Merge, concatenate, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed


def model_branched_cnn_mixed_lstm_regression_batchNorm(input1Shape, input2Shape, outputShape, numFilter=4,
                                                       numUnitLSTM=16, batchSize=None,
                                                       stateful=False,
                                                       dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'mean_squared_error'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]
    input1 = Input(batch_shape=(batchSize,) + input1Shape)
    input2 = Input(batch_shape=(batchSize,) + input2Shape)

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(BatchNormalization())
    branch1.add(Activation('relu'))

    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(BatchNormalization())

    branch1.add(Activation('relu'))
    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(BatchNormalization())

    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    convOutShape1 = branch1.layers[-1].output_shape
    branch1.add(Reshape((convOutShape1[1], np.prod(convOutShape1[2:4]))))
    #     integer multiple of output shape

    nPadTo = int(np.ceil(convOutShape1[1] / ntOut) * ntOut)
    nPadding = (nPadTo - convOutShape1[1])
    branch1.add(ZeroPadding1D(padding=(0, nPadding)))
    print('original size: ')
    print(convOutShape1[1])
    print('padding to multiples of:')
    print(ntOut)

    print('new shape: ')
    print(branch1.layers[-1].output_shape)
    branch1.add(Reshape((ntOut, -1)))
    branch1.add(Dropout(dropout))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    branch2.add(MaxPooling2D(pool_size=(1, 2)))
    branch2.add(BatchNormalization())
    branch2.add(Activation('relu'))
    branch2.add(Dropout(dropout))

    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    branch2.add(MaxPooling2D(pool_size=(1, 2)))
    branch2.add(BatchNormalization())
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    branch2.add(Dropout(dropout))
    output1 = branch1(input1)
    output2 = branch2(input2)

    #     model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    #     model.add(keras.layers.merge.concatenate([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1,branch2],axis=2))
    mergedInput = concatenate([output1, output2], axis=2)

    #     keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    #     model.add(LSTM(48, return_sequences=True, stateful=True))
    X = LSTM(numUnitLSTM, return_sequences=True, stateful=stateful)(mergedInput)
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)
    #     model.add(LSTM(48, return_sequences=True))
    X = LSTM(numUnitLSTM, return_sequences=True)(X)
    X = BatchNormalization()(X)
    X = Dropout(dropout)(X)

    #     model.add(TimeDistributed(Dense(1)))
    output = TimeDistributed(Dense(1))(X)
    #     model.add(Activation('sigmoid'))

    model = Model(inputs=[input1, input2], outputs=output)

    return model, optimizer, loss

def model_branched_cnn_mixed_lstm_regression_functional(input1Shape, input2Shape, outputShape, numFilter=4, batchSize=None,
                                                    stateful=False,
                                                    dropout=0):
    """model with mixed sampling rate and type
    input 1: (nsample, nt,nchannel,1) for time series
    input 2: (nsample, nt,nfreq,nchannel) for spectrogram
    """
    optimizer = 'adam'
    loss = 'mean_squared_error'
    kernelSize1 = (3, 3)
    kernelSize2 = (2, 5)
    ntOut = outputShape[0]
    input1 = Input(batch_shape=(batchSize,) + input1Shape)
    input2 = Input(batch_shape=(batchSize,) + input2Shape)

    # -------------branch 1 : time series -----------------
    branch1 = Sequential()
    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same', batch_input_shape=(batchSize,) + input1Shape))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))

    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    branch1.add(Dropout(dropout))

    branch1.add(Convolution2D(numFilter, kernelSize1, padding='same'))
    branch1.add(MaxPooling2D(pool_size=(2, 1)))
    branch1.add(Activation('relu'))
    #     reshape branch 1 input to match output time step: from (ntConv,nc,nfilter) to (ntOut,ntConv//ntConv*nc*nfilter)
    branch1.add(Reshape((ntOut, -1)))
    branch1.add(Dropout(dropout))

    # -------------branch 2 : spectrogram -----------------
    branch2 = Sequential()
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same', batch_input_shape=(batchSize,) + input2Shape))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    branch2.add(Dropout(dropout))
    branch2.add(Convolution2D(numFilter, kernelSize2, padding='same'))
    #     branch2.add(MaxPooling2D(pool_size=(2, 4)))
    branch2.add(Activation('relu'))
    convOutShape2 = branch2.layers[-1].output_shape
    branch2.add(Reshape((convOutShape2[1], np.prod(convOutShape2[2:4]))))

    branch2.add(Dropout(dropout))
    output1 = branch1(input1)
    output2 = branch2(input2)

    #     model = Sequential()

    # -------------merge branch 1 with branch 2 -----------------
    #     model.add(keras.layers.merge.concatenate([branch1, branch2], mode='concat', concat_axis=2))
    #     model.add(concatenate([branch1,branch2],axis=2))
    mergedInput = concatenate([output1, output2], axis=2)

    #     keras.layers.concatenate
    #     convOutShape = model.layers[-1].output_shape
    #     model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    #     model.add(Permute((2, 1)))
    #     model.add(LSTM(48, return_sequences=True, stateful=True))
    X = LSTM(48, return_sequences=True, stateful=stateful)(mergedInput)
    X=Dropout(dropout)(X)
    #     model.add(LSTM(48, return_sequences=True))
    X = LSTM(48, return_sequences=True)(X)
    X=Dropout(dropout)(X)

    #     model.add(TimeDistributed(Dense(1)))
    output = TimeDistributed(Dense(1))(X)
    #     model.add(Activation('sigmoid'))

    model = Model(inputs=[input1, input2], outputs=output)

    return model, optimizer, loss

# -----------------model from 2predict---------
def model_cnn_cat_mixed_lstm_2predict(input1Shape, input2Shape, numClasses, batchSize=None,stateful=False,
                             dropout=0):

    optimizer = 'adam'
    loss = 'categorical_crossentropy'

    branch1 = Sequential()
    branch1.add(Convolution2D(4, 1, 5, border_mode='same', batch_input_shape=(batchSize,)+input1Shape))
    branch1.add(MaxPooling2D(pool_size=(1, 5)))
    branch1.add(Activation('relu'))
    #branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(4, 1, 5, border_mode='same'))
    branch1.add(MaxPooling2D(pool_size=(1, 5)))
    branch1.add(Activation('relu'))
    #branch1.add(Dropout(0.2))
    branch1.add(Convolution2D(4, 1, 5, border_mode='same'))
    branch1.add(MaxPooling2D(pool_size=(1, 4)))
    branch1.add(Activation('relu'))
    #branch1.add(Dropout(0.2))

    branch2 = Sequential()
    branch2.add(Convolution2D(4, 3, 5, border_mode='same', batch_input_shape=(batchSize,)+input2Shape))
    branch2.add(MaxPooling2D(pool_size=(2, 5)))
    branch2.add(Activation('relu'))
    #branch2.add(Dropout(0.2))
    branch2.add(Convolution2D(4, 3, 5, border_mode='same'))
    branch2.add(MaxPooling2D(pool_size=(1, 2)))
    branch2.add(Activation('relu'))
    #branch2.add(Dropout(0.2))

    model = Sequential()
    model.add(Merge([branch1, branch2], mode='concat', concat_axis=2))

    convOutShape = model.layers[-1].output_shape
    model.add(Reshape((np.prod(convOutShape[1:3]), convOutShape[3])))
    model.add(Permute((2, 1)))
    model.add(LSTM(48, return_sequences=True, stateful=True))
    #model.add(Dropout(0.2))
    model.add(LSTM(48, return_sequences=True))
    #model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(numClasses)))
    model.add(Activation('softmax'))

    return model, optimizer, loss
