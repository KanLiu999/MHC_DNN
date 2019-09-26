# Add AUC calculation.
# Using keras sequential to build deep neural network to classify MHC 9-aa motif


#%%
# import packages
from __future__ import print_function
import numpy as np
import os, sys
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras import regularizers
from keras.optimizers import RMSprop, Adam, Adadelta, Adagrad
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasClassifier
from keras.metrics import binary_accuracy
from keras.callbacks import CSVLogger
from sklearn.metrics import auc, roc_curve, roc_auc_score, precision_recall_curve, f1_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

np.random.seed(456)  # for reproducibility


#%%
# Save the traning to log 
csv_logger = CSVLogger('/home/liukan/Projects/Script/Neural_network_code/Keras_test/DNN_Test_4_Keras_AUC_plot_using_cluster_2019_09_17.model_training.log', append=True, separator=';')


#########################
#%%
### Import train and test data
train_file1 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_cluster_encoded_2019_09_17.txt", delimiter='\t')
train_file2 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_cluster_extract_encoded_2019_09_17.txt", delimiter='\t')
train_file3 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton_encoded_2019_09_17.txt", delimiter='\t')[0:1720,::] 
train_file4 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract_encoded_2019_09_17.txt", delimiter='\t')[0:1720,::] 

test_file1 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton_encoded_2019_09_17.txt", delimiter='\t')[1720:,::]  
test_file2 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract_encoded_2019_09_17.txt", delimiter='\t')[1720:,::]

train_label1 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_cluster_label_2019_09_17.txt", delimiter='\t')
train_label2 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_cluster_extract_label_2019_09_17.txt", delimiter='\t')
train_label3 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton_label_2019_09_17.txt", delimiter='\t')[0:1720,::] 
train_label4 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract_label_2019_09_17.txt", delimiter='\t')[0:1720,::] 

test_label1 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/binder_singleton_label_2019_09_17.txt", delimiter='\t')[1720:,::] 
test_label2 = np.loadtxt("/home/liukan/Projects/All_projects/MHC_deep_network/Input_data/clustering/Sim_70/nonbinder_singleton_extract_label_2019_09_17.txt", delimiter='\t')[1720:,::]

train_total = np.concatenate((train_file1, train_file2, train_file3, train_file4), axis=0)
test_total = np.concatenate((test_file1, test_file2), axis=0)
train_total_label = np.concatenate((train_label1, train_label2, train_label3, train_label4), axis=0)
test_total_label = np.concatenate((test_label1, test_label2), axis=0)

merge_data =  np.concatenate((train_total, test_total), axis=0)
merge_label = np.concatenate((train_total_label, test_total_label), axis=0)
merge_label = merge_label[:,0]


#%%
# Remove the propensity related features to test the performance.

# Modified 2D Numpy Array by removing columns at given index range
train_total = np.delete(train_total,np.s_[2:21], axis=1)
test_total = np.delete(test_total,np.s_[2:21], axis=1)


## Shuffling the files
shuffle_temp = np.arange(train_total.shape[0])
np.random.shuffle(shuffle_temp)
train_total = train_total[shuffle_temp]
train_total_label = train_total_label[shuffle_temp][:,0]

shuffle_temp = np.arange(test_total.shape[0])
np.random.shuffle(shuffle_temp)
test_total = test_total[shuffle_temp]
test_total_label = test_total_label[shuffle_temp][:,0]




#%%
# Network and training parameter initialization & Build AUROC function

INPUT_DIM = 431 # Total number of input features shape.
NB_EPOCH = 150 # Should be no greater than 50 to avoid overfitting based on my experience.
BATCH_SIZE = 32 # 128, 100, 50 or 32 depending on your data size and preference. 
VERBOSE = 1 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
NB_CLASSES = 2   # Number of output categories. For MHC classification, 1 represents binder, 0 represents nonbinder.
OPTIMIZER = Adadelta() # Optimizer, which can be Adagrad, Adadelta, RMSprop, etc.
N_HIDDEN = 64 # Number of hidden layer in NN.
VALIDATION_SPLIT = 0.2 # How much train data is reserved for validation.
DROPOUT = 0.5 # Dropout rate for the sequential neurons. Usually from 0.5 to 0.1. 

# evaluate AUROC
def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


#%%
# Try model from Keras example from its website
# https://keras.io/getting-started/sequential-model-guide/

model = Sequential()
model.add(Dense(N_HIDDEN, input_dim=INPUT_DIM, kernel_initializer='uniform', kernel_constraint=maxnorm(4), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_HIDDEN, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(N_HIDDEN, kernel_initializer='uniform', kernel_regularizer=regularizers.l2(0.01),activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=OPTIMIZER,
              metrics=['accuracy', auroc])

history = model.fit(train_total, train_total_label, validation_split=VALIDATION_SPLIT, 
          epochs=NB_EPOCH,
          batch_size=BATCH_SIZE, callbacks=[csv_logger])
score = model.evaluate(test_total, test_total_label, batch_size=BATCH_SIZE)

print("\nTest score:", score[0])
print('Test accuracy:', score[1])



#%%
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#%%
# Define the roc curve, precision-recall curve.

# predict probabilities
y_pred_probs = model.predict_proba(test_total)
# keep probabilities for the positive outcome only
y_pred_probs = y_pred_probs.ravel()

# predict class values
yhat = model.predict(test_total).ravel()
# calculate precision-recall curve
precision, recall, thresholds = precision_recall_curve(test_total_label, y_pred_probs)
auc = auc(recall, precision)
# calculate average precision score
ap = average_precision_score(test_total_label, y_pred_probs)
# calculate F1 score
f1 = f1_score(test_total_label, np.round(yhat))

print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))

y_pred_keras = model.predict(test_total)[:,:1].ravel()
y_test = test_total_label.ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
# auc_keras = auc(fpr_keras, tpr_keras)


#%%
#Plot the ROC curve
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='Keras DNN (area = {:.3f})'.format(auc))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()


#%%
# Plot the precision and recall rate
plt.plot([0, 1], [0.1, 0.1], linestyle='--')
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall(Sensitivity)')
plt.ylabel('Precision(Positive predictive value)')
plt.title('Precision Recall curve')
# show the plot
plt.show()


#%%
model.save('/home/liukan/Projects/Script/Neural_network_code/Keras_test/DNN_Test_4_Keras_AUC_plot_using_cluster_2019_09_17.model.h5')


