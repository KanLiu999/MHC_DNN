#%%
# Import modules and functions
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

np.random.seed(123)

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
## Shuffling the files
shuffle_temp = np.arange(train_total.shape[0])
np.random.shuffle(shuffle_temp)
train_total = train_total[shuffle_temp]
train_total_label = train_total_label[shuffle_temp][:,0]

shuffle_temp = np.arange(test_total.shape[0])
np.random.shuffle(shuffle_temp)
test_total = test_total[shuffle_temp]
test_total_label = test_total_label[shuffle_temp][:,0]

feature_size = len(train_total[0])

#%%
# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors = 15)
# Fit the classifier to the data
knn.fit(train_total,train_total_label)

#show first 5 model predictions on the test data
knn.predict(test_total)[0:5]

#check accuracy of our model on the test data
knn.score(test_total, test_total_label)

#%%
#create a new KNN model
knn_cv = KNeighborsClassifier(n_neighbors=15)
#train model with cv of 5 
cv_scores = cross_val_score(knn_cv, merge_data, merge_label, cv=5)
#print each cv score (accuracy) and average them
print(cv_scores)
print('Mean of 5 fold cross validation scores: {}'.format(np.mean(cv_scores)))

#%%

#create new a knn model
knn2 = KNeighborsClassifier()
#create a dictionary of all values we want to test for n_neighbors
param_grid = {'n_neighbors': np.arange(1, 25)}
#use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
#fit model to data
knn_gscv.fit(merge_data, merge_label)

#check top performing n_neighbors value
knn_gscv.best_params_
# best n_neighbor: 15
