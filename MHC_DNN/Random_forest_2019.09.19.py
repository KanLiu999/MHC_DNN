#%%
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

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
# Run the Random Forest
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 2,
              'max_features': 'auto',
              'max_depth': 10,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)

RF_model.fit(train_total, train_total_label)
RF_predictions = RF_model.predict(test_total)
score = accuracy_score(test_total_label ,RF_predictions)
print('Accuracy:', round(score*100, 8), '%.')
errors = abs(RF_predictions - test_total_label)
print('Average absolute error:', round(np.mean(errors), 2), 'degrees.')



#%%
# Calculate feature importance
importances = list(RF_model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(np.arange(feature_size), importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:10} Importance: {}'.format(*pair)) for pair in feature_importances if pair[1]>0]

#%%
# list of x locations for plotting
x_values = list(range(len(importances)))
important_dictionary = dict(zip(x_values, importances))

# print('Dictionary in descending order by value : ', sorted_dictionary)

feature_count = 30

important_feature_ID = []
important_feature_value = []
for k,v in sorted(important_dictionary.items(), key=operator.itemgetter(1), reverse=True)[:feature_count]:
    important_feature_ID.append(k)
    important_feature_value.append(v)

#%%
my_order = np.arange(len(important_feature_ID))

# Make a bar chart
plt.bar(my_order, important_feature_value, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(my_order, important_feature_ID, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance')
plt.xlabel('Feature ID')
plt.title('Most important '+str(feature_count)+' features')
plt.savefig('/home/liukan/Projects/All_projects/MHC_deep_network/Plot/feature_importance.pdf')


#%%
# List of features sorted from most to least important
# Cumulative importances
cumulative_importances = np.cumsum(important_feature_value)
# Make a line graph
plt.plot(my_order, cumulative_importances, 'g-')
# Draw line at 95%of importance retained
# plt.hlines(y = 0.95, xmin=0, xmax=len(important_feature_value), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(my_order, important_feature_ID, rotation = 'vertical')

# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative importance');




#%%
