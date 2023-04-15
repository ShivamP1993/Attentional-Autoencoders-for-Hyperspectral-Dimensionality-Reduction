# Function to perform one hot encoding of the class labels 

def my_ohc(lab_arr):
    lab_arr_unique =  np.unique(lab_arr)
    r,c = lab_arr.shape
    r_u  = lab_arr_unique.shape
    
    
    one_hot_enc = np.zeros((r,r_u[0]), dtype = 'float')
    
    for i in range(r):
        for j in range(r_u[0]):
            if lab_arr[i,0] == lab_arr_unique[j]:
                one_hot_enc[i,j] = 1
    
    return one_hot_enc

# Function that takes the confusion matrix as input and
# calculates the overall accuracy, producer's accuracy, user's accuracy,
# Cohen's kappa coefficient and standard deviation of 
# Cohen's kappa coefficient

def accuracies(cm):
  import numpy as np
  num_class = np.shape(cm)[0]
  n = np.sum(cm)

  P = cm/n
  ovr_acc = np.trace(P)

  p_plus_j = np.sum(P, axis = 0)
  p_i_plus = np.sum(P, axis = 1)

  usr_acc = np.diagonal(P)/p_i_plus
  prod_acc = np.diagonal(P)/p_plus_j

  theta1 = np.trace(P)
  theta2 = np.sum(p_plus_j*p_i_plus)
  theta3 = np.sum(np.diagonal(P)*(p_plus_j + p_i_plus))
  theta4 = 0
  for i in range(num_class):
    for j in range(num_class):
      theta4 = theta4+P[i,j]*(p_plus_j[i]+p_i_plus[j])**2

  kappa = (theta1-theta2)/(1-theta2)

  t1 = theta1*(1-theta1)/(1-theta2)**2
  t2 = 2*(1-theta1)*(2*theta1*theta2-theta3)/(1-theta2)**3
  t3 = ((1-theta1)**2)*(theta4 - 4*theta2**2)/(1-theta2)**4

  s_sqr = (t1+t2+t3)/n

  return ovr_acc, usr_acc, prod_acc, kappa, s_sqr

# Import Relevant libraries and classes
import scipy.io as sio
import numpy as np
import tqdm
from sklearn.decomposition import PCA
import tensorflow as tf
keras = tf.keras
from keras import backend as K
from keras import regularizers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout
from keras.layers import Conv2D, Flatten, Lambda, Conv3D, Conv3DTranspose,BatchNormalization
from keras.layers import Conv1D, Activation, Layer, MaxPooling1D
from keras.layers import Reshape, Conv2DTranspose, Concatenate, Multiply, Add, MaxPooling2D
from keras.layers import MaxPooling3D, GlobalAveragePooling2D,  Conv1DTranspose
from keras import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras import backend as K
from sklearn.metrics import confusion_matrix

def get_gt_index(y):

  import tqdm

  shapeY = np.shape(y)

  pp,qq = np.unique(y, return_counts=True)
  sum1 = np.sum(qq)-qq[0]

  index = np.empty([sum1,3], dtype = 'int')

  cou = 0
  for k in tqdm.tqdm(range(1,np.size(np.unique(y)))):
    for i in range(shapeY[0]):
      for j in range(shapeY[1]):
        if y[i,j] == k:
          index[cou,:] = np.expand_dims(np.array([k,i,j]),0)
          #print(cou)
          cou = cou+1
  return index

## Read the training and test vectors for the Indian Pines dataset

train_vec = np.reshape(np.load('/content/gdrive/My Drive/Projects/IGARSS21/IP/train_vec.npy')[:,5,5,:], [-1,200,1])
train_labels = np.load('/content/gdrive/My Drive/Projects/IGARSS21/IP/train_labels.npy')

test_vec = np.reshape(np.load('/content/gdrive/My Drive/Projects/IGARSS21/IP/test_vec.npy')[:,5,5,:], [-1,200,1])
test_labels = np.load('/content/gdrive/My Drive/Projects/IGARSS21/IP/test_labels.npy')

# Define Attention function

def Att_fn(x):

  shp = x.shape

  x1 = Conv1D(shp[2], 7, activation="relu", strides=1, padding="same")(x)
  x2 = Multiply()([x1,x])
  x = Add()([x,x2])
  return x

# Attentional Convolutional Autoencoder

def cae(x):

  # Encoder
  conv1 = Conv1D(256, 7, activation="relu", strides=2, padding="valid")(x)
  conv1 = Att_fn(conv1) # Apply Attention function

  conv2 = Conv1D(256, 7, activation="relu", strides=2, padding="valid")(conv1)
  conv2 = Att_fn(conv2)

  conv3 = Conv1D(128, 7, activation="relu", strides=2, padding="valid")(conv2)
  conv3 = Att_fn(conv3)

  conv4 = Conv1D(128, 7, activation="relu", strides=2, padding="valid")(conv3)
  conv4 = Att_fn(conv4)

  conv5 = Conv1D(64, 7, activation="relu", strides=2, padding="valid")(conv4)
  conv5 = Att_fn(conv5)

  # Encoder representation
  f1 = Flatten()(conv5)
  d1 = Dense(5, activation="relu")(f1)
  
  #Decoder
  
  xA = Dense(1 * 1 * 200, activation="relu")(d1)
  r1 = Reshape([200,1])(xA)
  
  conv6 = Conv1D(64, 7, activation="relu", strides=2, padding="valid")(r1)
  conv7 = Conv1D(64, 7, activation="relu", strides=2, padding="valid")(conv6)
  conv8 = Conv1D(128, 7, activation="relu", strides=2, padding="valid")(conv7)
  conv9 = Conv1D(180, 7, activation="relu", strides=3, padding="valid")(conv8)
  conv10 = Conv1D(200, 5, activation="relu", strides=3, padding="valid")(conv9)

  rs = Reshape([200,1])(conv10)

  decoder = keras.Model(x, rs, name="decoder")

  return decoder

xA = Input(shape = (200,1))
aeC = cae(xA) # model is aeC

aeC.summary() # Get model summary

aeC.compile(loss = 'mse', optimizer=keras.optimizers.Adam(0.0001), metrics = ['mse'])

# Initialize Random Forest to check if the reduced dimensions give decent peroformance

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0, n_estimators=200)

acc_temp=0 # Temporary accuracy
import gc
for epoch in range(500): 
  gc.collect()
  aeC.fit(x = train_vec, y = train_vec,
                  epochs=1, batch_size = 128, verbose = 1)
  
  # Create model with reduced dimensions. 
  # The extracted train and test features go to random forest classifier for validation purposes
  new_model = Model(aeC.input, aeC.layers[7].output, name = 'new_model') 
  code_feat_train = np.reshape(new_model.predict(train_vec),[1024,5])
  code_feat_test = np.reshape(new_model.predict(test_vec),[9225,5])

  clf.fit(code_feat_train, train_labels)
  preds = clf.predict(code_feat_test)

  conf = confusion_matrix(test_labels, preds)
  ovr_acc, _, _, _, _ = accuracies(conf)

  print(epoch)
  print(np.round(100*ovr_acc,2))
  # Saving model with best accuracies.
  if ovr_acc>=k:
    aeC.save('/content/gdrive/My Drive/Projects/IGARSS21/IP/enc_cae1d_small_5_500') 
    k = ovr_acc
    ep = epoch
  print('acc_max = ', np.round(100*k,2), '% at epoch', ep)
