#%%
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

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
## Shuffling the data
shuffle_temp = np.arange(train_total.shape[0])
np.random.shuffle(shuffle_temp)
train_total = train_total[shuffle_temp]
train_total_label = train_total_label[shuffle_temp][:,0]

shuffle_temp = np.arange(test_total.shape[0])
np.random.shuffle(shuffle_temp)
test_total = test_total[shuffle_temp]
test_total_label = test_total_label[shuffle_temp][:,0]

feature_size = len(train_total[0])




tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

