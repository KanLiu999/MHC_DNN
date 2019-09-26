#%%
# import modules
import tensorflow as tf
import numpy as np
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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

# Shuffling the data
shuffle_temp = np.arange(train_total.shape[0])
np.random.shuffle(shuffle_temp)
train_total = train_total[shuffle_temp]
train_total_label = train_total_label[shuffle_temp][:,0]

shuffle_temp = np.arange(test_total.shape[0])
np.random.shuffle(shuffle_temp)
test_total = test_total[shuffle_temp]
test_total_label = test_total_label[shuffle_temp][:,0]

# The size of training data
feature_size = len(train_total[0])

#%%
# grid search
lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=learning_rate, max_features=3, max_depth=3, random_state=0)
    gb_clf.fit(train_total, train_total_label)

    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(train_total, train_total_label)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(test_total, test_total_label)))
    predictions = gb_clf.predict(test_total)
    print("Confusion Matrix:")
    print(confusion_matrix(test_total_label, predictions))
    print("Classification Report")
    print(classification_report(test_total_label, predictions))
    print('*'*30)



#%%
