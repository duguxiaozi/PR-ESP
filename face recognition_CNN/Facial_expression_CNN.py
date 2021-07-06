import numpy as np
# import tensorflow as tf
from keras import layers
import time
import keras
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn import preprocessing
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import os, sys
#import mat4py
from scipy.io import loadmat
import keras.backend as K
K.set_image_data_format( 'channels_last' )
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from csv import reader
import numpy as np

#使用标准的Python类库导入csv数据
# filename = 'faceDR.csv'
# with open( filename, 'rt' ) as raw_data:	#导入标签数据
#     readers = reader( raw_data, delimiter=',' )
#     x = list( readers )
#     Xtraindata = np.array( x )
# print(Xtraindata.shape)
#
# filename = 'faceDS.csv'
# with open( filename, 'rt' ) as raw_data:
#     readers = reader( raw_data, delimiter=',' )
#     x = list( readers )
#     Xtestdata = np.array( x )
# print(Xtestdata.shape)
#
# Xlabel = np.vstack((Xtraindata,Xtestdata))
# print(Xlabel.shape)
# np.save('Xlabel.npy', Xlabel)
#
# j = 0
# Xtrain = np.ones( (3991, 128, 128) )
# # Xtrain = []  # 设立训练集列表
# for i in range( 1223, 5222, 1 ):
# 	if i == 1228 or i == 1232 or i == 1808 or i == 4056 or i == 4135 or i == 4136 or i == 5004 or i == 2412 or i == 2416:
# 		continue
# 	data = loadmat( 'rawdatamat/' + str( i ) + '.mat' )
# 	image = data['I']
# 	print("第%d张图片"%(i))
# 	print( image.shape )
# 	images = image.reshape( 128, 128 )
# 	Xtrain[j, :, :] = images
# 	j = j + 1
# np.save('Xtrain.npy', Xtrain)

Xdata = np.load('../Xtrain.npy')	# 导入图像数据
Xlabel = np.load('../Xlabel.npy')  # 导入图像标签

Xmood = Xlabel[:,[8]]   #第8为提取表情标签
Xmoodlabel = np.zeros((3991,3))
#print( Xmoodlabel )
for i in range(3991):
    a = int(Xmood[i,:])   #字符转化整型
    if a==0:
        Xmoodlabel[i, 0] = 1
    if a==1:
        Xmoodlabel[i, 1] = 1
    if a==3:
        Xmoodlabel[i, 2] = 1

# Xsex = Xlabel[:,[2]]   #第2为提取性别标签
# Xsexlabel = np.ones((3991,1))
# #print( Xmoodlabel )
# for i in range(3991):
#      a = int(Xsex[i,:])   #字符转化整型
#      Xsex[i, :] = a

# Xtrain = Xdata[0:3550,:,:]
# Ytrain = Xsexlabel[0:3550,:]
# Xtest = Xdata[3551:3990,:,:]
# Ytest = Xsexlabel[3551:3990,:]

Xtrain = Xdata[0:3550,:,:]
Ytrain = Xmoodlabel[0:3550,:]
Xtest = Xdata[3551:3990,:,:]
Ytest = Xmoodlabel[3551:3990,:]

#GRADED FUNCTION: HappyModel

def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    ### START CODE HERE ###
    X_input = Input(shape=input_shape)
    X = ZeroPadding2D(padding=(1, 1))(X_input)
    X = Conv2D(8, kernel_size=(2,2), strides=(1,1))(X)#8个filter
    X = BatchNormalization(axis=3)(X)#
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)
    #
    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(16, kernel_size=(2,2), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

    X = ZeroPadding2D(padding=(1, 1))(X)
    X = Conv2D(32, kernel_size=(2,2), strides=(1,1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')(X)

    # FC
    X = Flatten()(X)
    Y = Dense(3, activation='sigmoid')(X)#就是全连接层

    model = Model(inputs = X_input, outputs = Y, name='HappyModel')
    ### END CODE HERE ###

    return model

kf = KFold(n_splits = 10, random_state=2001, shuffle=True)#k折交叉验证

scores = []
ind = 0
for train_index,valid_index in kf.split(Xdata):
     happyModel = HappyModel( (128,128,1) )
     start = time.time()
     happyModel.compile(
          optimizer=keras.optimizers.Adam( lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0 ),
          loss='binary_crossentropy', metrics=['accuracy'] )
     ### END CODE HERE ###
     stop = time.time()
     print( "compile step cost=", str( stop - start ) )

     start = time.time()
     happyModel.fit( x=Xdata[train_index], y=Xmoodlabel[train_index], batch_size=20, epochs=7)  # 模型训练
     stop = time.time()
     preds = happyModel.evaluate( x=Xdata[valid_index], y=Xmoodlabel[valid_index] )  # 模型测试
     score = preds[1]
     scores.append( score )
     ind = ind+1
     print( '%d : current Acc: %.3f' % (ind,score) )
     print( "fit step cost=", str( stop - start ) )

print( '\nCV accuracy: %.3f ' % (np.mean( scores ) ))


### END CODE HERE ###


print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print("test step cost=",str(stop-start))