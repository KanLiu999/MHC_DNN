# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 09:03:24 2020

@author: m203318
"""

'''
  Visualizing how layers represent classes with keras-vis Saliency Maps.
'''
# =============================================
# Plot ECG
# =============================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
l=['I','II','III','AvR','Avl','AvF','V1','V2','V3','V4','V5','V6']
l2=['II','V1','V5']

def plotMuse(curECG,fname='//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/paper/EHJ/reviewrs/example12.png',grid=True):
    newECG = np.zeros([5000,6])
    
    
    newECG[0:1250,0] = curECG[:1250,0] # I
    newECG[1250:2500,0] = curECG[1250:2500,3] # R
    newECG[2500:3750,0] = curECG[2500:3750,6] # 1
    newECG[3750:,0] = curECG[3750:,9] # 1
    
    
    newECG[0:1250,1] = curECG[:1250,1] # II
    newECG[1250:2500,1] = curECG[1250:2500,4] # L
    newECG[2500:3750,1] = curECG[2500:3750,7] # 2
    newECG[3750:,1] = curECG[3750:,10] # 2
    
    newECG[0:1250,2] = curECG[:1250,2] # III
    newECG[1250:2500,2] = curECG[1250:2500,5] # F
    newECG[2500:3750,2] = curECG[2500:3750,8] # 3
    newECG[3750:,2] = curECG[3750:,11] # 3
    
    
    newECG[:,3] = curECG[:,1] # II
    newECG[:,4] = curECG[:,6] # V2
    newECG[:,5] = curECG[:,10] # V5
    
    
    newECG[::1250,:3]=300
    
    plt.cla()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(14.5, 10.5)
    
#Labels:
    for i in range(12):    
        x_text = 2.5 * (i//3) 
        y_text = -2000*(i%3)+300
        plt.text(x_text,y_text,l[i],fontsize=12)

    for i in range(3):    
        plt.text( -.2,-2000*(i+3)+100,l2[i],fontsize=12)
        
    pilotX = np.array([0,0.04,0.04,.16,.16,0.2])-.4
    pilotY= np.array([0,0,1000,1000,0,0])
        
        
    if grid:
        for j in range(-11500,1500+1,100):
            if j%500==0:
                lw  = 0.5
            else:
                lw = 0.1
            plt.plot([-.4,10],[j,j],color='red',linewidth=lw);
    
        for j in range(-10,251):
            if j%5==0:
                lw  = 0.5
            else:
                lw = 0.1
            plt.plot([j*.2/5,j*.2/5],np.array([-11500,1500]),color='red',linewidth=lw)
    plt.axis('off')
    for i in range(6):
        plt.plot(pilotX, pilotY - 2000*i ,color='black',linewidth=1)
        plt.plot(np.linspace(0,10,5000),newECG[:,i]-2000*i,color='black',linewidth=.8);

    #plt.savefig(fname, bbox_inches='tight',dpi=300)


def plotMuse2(curECG, color='blue', fname='//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/paper/EHJ/reviewrs/example12.png',grid=True):
    newECG = np.zeros([5000,6])
    
    
    newECG[0:1250,0] = curECG[:1250,0] # I
    newECG[1250:2500,0] = curECG[1250:2500,3] # R
    newECG[2500:3750,0] = curECG[2500:3750,6] # 1
    newECG[3750:,0] = curECG[3750:,9] # 1
    
    
    newECG[0:1250,1] = curECG[:1250,1] # II
    newECG[1250:2500,1] = curECG[1250:2500,4] # L
    newECG[2500:3750,1] = curECG[2500:3750,7] # 2
    newECG[3750:,1] = curECG[3750:,10] # 2
    
    newECG[0:1250,2] = curECG[:1250,2] # III
    newECG[1250:2500,2] = curECG[1250:2500,5] # F
    newECG[2500:3750,2] = curECG[2500:3750,8] # 3
    newECG[3750:,2] = curECG[3750:,11] # 3
    
    
    newECG[:,3] = curECG[:,1] # II
    newECG[:,4] = curECG[:,6] # V2
    newECG[:,5] = curECG[:,10] # V5
    
    
    newECG[::1250,:3]=300
    
    #plt.cla()
    #fig = matplotlib.pyplot.gcf()
    #fig.set_size_inches(14.5, 10.5)
    
    for i in range(6):
        #plt.plot(pilotX, pilotY - 2000*i ,color='black',linewidth=1)
        plt.plot(np.linspace(0,10,5000),newECG[:,i]-2000*i,color=color,linewidth=.8);
    plt.savefig(fname, bbox_inches='tight',dpi=300)
    #plt.savefig("temp.pdf")

   


# =============================================
# Model to be visualized
# =============================================
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras import activations

# Model configuration
img_width, img_height = 28, 28
batch_size = 250
no_epochs = 25
no_classes = 10
validation_split = 0.2
verbosity = 1

# Load MNIST dataset
(input_train, target_train), (input_test, target_test) = mnist.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0], 1, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 1, img_width, img_height)
    input_shape = (1, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 1)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 1)
    input_shape = (img_width, img_height, 1)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax', name='visualized_layer'))

# Compile the model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

model_minst = model

# =============================================
# Saliency Maps code
# =============================================
from vis.visualization import visualize_saliency
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)  

# Numbers to visualize
indices_to_visualize = [ 0, 12, 38]

# Visualize
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = input_test[index_to_visualize]
  input_class = np.argmax(target_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 2)
  # Generate visualization
  visualization = visualize_saliency(model, layer_index, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0]) 
  axes[0].set_title('Original image')
  axes[1].imshow(visualization)
  axes[1].set_title('Saliency map')
  fig.suptitle(f'MNIST target = {input_class}')
  plt.show()
  
# =============================================
# Saliency Maps code - for AS model
# =============================================
from vis.visualization import visualize_saliency
from vis.visualization import get_num_filters
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from keras import activations
from vis.visualization import visualize_cam
from vis.visualization import visualize_activation

#load AS model
# model_as = load_model('Z:/Michal/models_rename/AS.h5',compile=False)
# model_as = load_model('//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/dataset_1989_to_2019_last_echo/above_18/new_label/1st_TTE/results/ECG_Age_Sex/Single_lead/lead8/Combine_AS_2021-01-10_21%3A38%3A54.h5',compile=False)
model_as = load_model('/Volumes/remote-monitoring/Michal/models_rename/AS.h5')
#model.summary()
# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model_as, 'fc')
#filter_num = get_num_filters(layer_index)
# Swap sigmoid with linear
model_as.layers[layer_index].activation = activations.linear
#model_as = utils.apply_modifications(model_as)  
model_as.compile('adam', loss='binary_crossentropy')

# Numbers to visualize
#indices_to_visualize = [ 0, 12, 38, 83]

input_test_as = np.load('//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/dataset_1989_to_2019_last_echo/above_18/new_label/1st_TTE/dataset/test_1st_TTE_chunk_1_ECGs_12_lead_test_x_01262020_1550.npy')
target_test_as = np.load('//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/dataset_1989_to_2019_last_echo/above_18/new_label/1st_TTE/dataset/test_y_01262020_2120.npy')
# print ECG
#plotMuse(curECG=np.squeeze(input_test_as[1], axis=2))

indices_to_visualize = [72164]
#index_to_visualize = TP - 28074 , 8317, 4947, TN- 91992, FN - 101897, 38157, 72164 FP- 26520, 91331
# Visualize
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image_as = input_test_as[index_to_visualize]
  #input_image_as = np.squeeze(input_image_as, axis=2)
  #input_class_as = np.argmax(target_test_as[index_to_visualize])
  input_class_as = target_test_as[index_to_visualize,0]
  # Class object
  classes = {
    0: 'Normal',
    1: 'AS positive'
    }
  input_class_name = classes[input_class_as]
  #plt.plot(input_image)
  #fig, (ax1) = plt.subplots(1, 1, sharey=True)
 
   # Generate visualization - Grad-CAM
  #visualization_grad_cam = visualize_cam(model_as, layer_index, filter_indices=input_class_as, seed_input=input_image_as)

  # Generate visualization - Activation Maximization
  #visualization_max_act = visualize_activation(model_as, layer_index, filter_indices=1, input_range=(0., 1.))
  #visualization_max_act1 = visualization_max_act[:,:,0]
  # Generate visualization
  visualization_as = visualize_saliency(model_as, layer_index, filter_indices=input_class_as, seed_input=input_image_as)
  #plt.plot(visualization_as)
  #fig.suptitle(f'Original ECG - Saliency map, AS target = {input_class_name}')
  x=visualization_as.copy()
  x[x<0.01]= np.nan
  
  #Grad-CAM
  #y=visualization_grad_cam.copy()
  #y[y<0.01]= np.nan
  
  print(f'Original ECG - Saliency map, AS target = {input_class_name}')
  
  plotMuse(curECG=np.squeeze(input_image_as, axis=2));plotMuse2(curECG=x*2000, color='blue')

  #plotMuse(curECG=np.squeeze(input_image_as, axis=2));plotMuse2(curECG=x*2000, color='red');plotMuse2(curECG=y*1500)
 # plt.savefig(fname='//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/paper/EHJ/reviewrs/vis_example12_Saliency_GRAND_CAM_2514655.png', bbox_inches='tight',dpi=300)

 plt.savefig(fname="//mfad/researchmn/Programs/REMOTE-MONITORING/AS_michal/dataset_1989_to_2019_last_echo/above_18/new_label/1st_TTE/results/full_info/AS_ECG_SaliencyMap_ID_8317_MRN_821767.pdf")
# =============================================
# Grad-CAM code
# =============================================
from vis.visualization import visualize_cam, overlay
from vis.utils import utils
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')
# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model)

indices_to_visualize = [0,12]

# Visualize
for index_to_visualize in indices_to_visualize:
  # Get input
  input_image = input_test[index_to_visualize]
  input_class = np.argmax(target_test[index_to_visualize])
  # Matplotlib preparations
  fig, axes = plt.subplots(1, 3)
  # Generate visualization
  visualization = visualize_cam(model, layer_index, filter_indices=input_class, seed_input=input_image)
  axes[0].imshow(input_image[..., 0], cmap='gray') 
  axes[0].set_title('Input')
  axes[1].imshow(visualization)
  axes[1].set_title('Grad-CAM')
  heatmap = np.uint8(cm.jet(visualization)[..., :3] * 255)
  original = np.uint8(cm.gray(input_image[..., 0])[..., :3] * 255)
  axes[2].imshow(overlay(heatmap, original))
  axes[2].set_title('Overlay')
  fig.suptitle(f'MNIST target = {input_class}')
  plt.show()
  
# =============================================
# Activation Maximization code
# =============================================
from vis.visualization import visualize_activation
from vis.utils import utils
import matplotlib.pyplot as plt

# Find the index of the to be visualized layer above
layer_index = utils.find_layer_idx(model, 'visualized_layer')

# Swap softmax with linear
model.layers[layer_index].activation = activations.linear
model = utils.apply_modifications(model) 

# Numbers to visualize
numbers_to_visualize = [ 0, 1, 2]

# Visualize
for number_to_visualize in numbers_to_visualize:
  visualization = visualize_activation(model, layer_index, filter_indices=number_to_visualize)
  plt.imshow(visualization[..., 0])
  plt.title(f'MNIST target = {number_to_visualize}')
  plt.show()