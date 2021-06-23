#set random seeds
from os import environ
environ['PYTHONHASHSEED'] = str(50)
from numpy.random import seed
seed(50)
import random as rn
rn.seed(50)
#wanddb code disabled
environ["WANDB_DISABLE_CODE"] = "true"
## import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
import random
import json
from sklearn.utils import shuffle
from datetime import datetime
from argparse import ArgumentParser
import itertools
import logging
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression,TheilSenRegressor,RANSACRegressor,HuberRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from scipy import stats
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

#import mlxtend
#from mlxtend.classifier import EnsembleVoteClassifier

# # Function for splitting training and test set
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


# Function for creating model pipelines
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline

# For standardization
from sklearn.preprocessing import StandardScaler

# Helper for cross-validation
from sklearn.model_selection import GridSearchCV

# Classification metrics (added later)
from sklearn.metrics import roc_curve, auc, roc_auc_score, recall_score
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

#test
##################################################Global params###################################################
wandb_enable = False

##################################################dataset functions###############################################
def set_random_seed(seed_num):
    print("do nothing")
    # random.seed(seed_num)
    # os.environ['PYTHONHASHSEED'] = str(seed_num)
    # np.random.seed(
    #     seed_num)  # TODO: try different seed numbers and check AUC stay the same, out this number to an arg'/config file


def split_group_by_prec(df, prec_num, seed_num):
    #set_random_seed(seed_num)
    msk = np.random.rand(len(df)) < prec_num
    train = df[msk]
    validation = df[~msk]
    return train, validation


def data_load(fileName):
    """Load the dataset and print the row# and column#."""
    df = pd.read_csv(fileName)
    df.head(3)
    logger.info(f"{df.shape[0]} samples and {df.shape[1]} features.")
    return df


def norm_fields(fields,train,test):
    """"Normelaize required fields"""
    for field in fields:
        train[field+'_N'] = (train[field]-np.mean(train[field]))/np.std(train[field])
        test[field+'_N'] = (test[field]-np.mean(train[field]))/np.std(train[field])
    return train, test


def create_train_test(df_pos, df_neg, features, target, time_f, path, norm_cont_var_list, seed_num, name):
    train_p, test_p = split_group_by_prec(df=df_pos, prec_num=0.70, seed_num=seed_num)
    train_n, test_n = split_group_by_prec(df=df_neg, prec_num=0.70, seed_num=seed_num)

    train = train_p.append(train_n)
    test = test_p.append(test_n)

    train = shuffle(train)
    test = shuffle(test)

    # add column group 1-train, 2-test
    train["group"] = 1
    test["group"] = 2

    # print positive negative balance and save dataset file
    print_dataset_numbers_and_save(train, test, time_f, path, target)

    # norm continues fields
    train, test = norm_fields(norm_cont_var_list, train, test)

    # define labels
    train_y = train[[target]]
    test_y = test[[target]]

    train_x = train[features]
    test_x = test[features]

    #save to file
    train_x.to_csv(path+name+'_df_train_x_'+time_f+'.csv', index=False)
    train_y.to_csv(path+name+'_df_train_y_'+time_f+'.csv', index = False)
    test_x.to_csv(path+name+'_df_test_x_'+time_f+'.csv', index = False)
    test_y.to_csv(path+name+'_df_test_y_'+time_f+'.csv', index = False)

    return train_x, train_y, test_x, test_y, train, test


def print_dataset_numbers_and_save(train, test, time_f, path, target):
    #save to a file in order to analyse results later
    df_all_groups = train.append(test)
    df_all_groups.to_csv(path+"all_group_dataset_balance_"+time_f+".csv", index=False)
    logger.info(f"Total dataset samples: {df_all_groups.shape[0]} ,balance:")
    logger.info(df_all_groups[target].value_counts())
    logger.info(f"Total train samples: {train.shape[0]} ,balance:")
    logger.info(train[target].value_counts())
    logger.info(f"Total test samples:  {test.shape[0]},balance:")
    logger.info(test[target].value_counts())

#####################################################CI 95% AUC functions########################################
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2

def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight)


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(predictions_sorted_transposed[r, :], sample_weight)
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m]*(tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1. - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count, ordered_sample_weight)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov

def print_AUC_CI(y_true, y_pred, alpha=.95):
    print('in print_AUC_CI function')
    print(f'y_true shape is:{y_true.shape}')
    print(f'y_pred shape is:{y_pred.shape}')
    auc, auc_cov = delong_roc_variance(
        y_true,
        y_pred)

    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

    ci = stats.norm.ppf(
        lower_upper_q,
        loc=auc,
        scale=auc_std)

    ci[ci > 1] = 1
    # print('AUC:', auc)
    # # print('AUC COV: '%auc_cov)
    # print(' AUC CI:%.3f-%.3f'% tuple(ci))
    return round(auc, 3), tuple(np.round(ci, 3))

def cur_calc_performance(test_y, test):
    fpr, tpr, th, roc_auc, optimalIndex = getRocData(test_y.values, test[['estimated10yriskhardascvd_pct']].values)

    est_y_binary = test[['estimated10yriskhardascvd_pct']].values >= 10
    cm = confusion_matrix(test_y.values[:, 0], est_y_binary)
    logger.info("Metrics on Testing dataset by current calculator - using 10% Th: ")
    logger.info(cm)
    tn, fp, fn, tp = confusion_matrix(test_y.values[:, 0], est_y_binary).ravel()
    logger.info(f"tn,fp,fn,tp {tn, fp, fn, tp}")
    accuracy = round((tn + tp) / (tp + tn + fp + fn), 3)
    sensitivity = round(tp / (tp + fn), 3)
    specificity = round(tn / (tn + fp), 3)
    ppv = round(tp / (tp + fp), 3)
    npv = round(tn / (tn + fn), 3)
    # adding 95% confidence interval AUC , df_est_test[1].values
    auc_ci = print_AUC_CI(test_y.values[:, 0], test[['estimated10yriskhardascvd_pct']].values[:, 0])

    return roc_auc, auc_ci, accuracy, sensitivity, specificity, npv, ppv
##############################################Imbalance strategy functions#######################################
def over_sample_func(train_x, train_y, target):
    try:
        logger.info(f"counter before oversample is: {train_y[target].value_counts()}")
        # transform the dataset
        oversample = SMOTE()
        train_x, train_y = oversample.fit_resample(train_x, train_y)
        # summarize the new class distribution
        logger.info(f"counter after oversample is: {train_y[target].value_counts()}")
        return train_x, train_y
    except Exception as ex:
        logger.error(f"failed to run over_sample_func due to: {ex}")


def over_under_sample_func(train_x, train_y, target):
    try:
        logger.info(f"counter before over_under_sample is: {train_y[target].value_counts()}")
        # transform the dataset
        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [('o', over), ('u', under)]
        pipeline = Pipeline(steps=steps)
        # transform the dataset
        train_x, train_y = pipeline.fit_resample(train_x, train_y)
        # summarize the new class distribution
        logger.info(f"counter after over_under_sample is: {train_y[target].value_counts()}")
        return train_x, train_y
    except Exception as ex:
        logger.error(f"failed to run over_under_sample_func due to: {ex}")


def borderline_smoth_func(train_x, train_y, target):
    try:
        logger.info(f"counter before border line SMOTH is: {train_y[target].value_counts()}")
        # transform the dataset
        #oversample = BorderlineSMOTE()
        oversample = SVMSMOTE()
        train_x, train_y = oversample.fit_resample(train_x, train_y)
        # summarize the new class distribution
        logger.info(f"counter after borderline SMOTH is: {train_y[target].value_counts()}")
        return train_x, train_y
    except Exception as ex:
        logger.error(f"failed to run borderline_smoth_func due to: {ex}")


def adaptive_synthetic_sampling_func(train_x, train_y, target):
    try:
        logger.info(f"counter before ADASYN is: {train_y[target].value_counts()}")
        # transform the dataset
        oversample = ADASYN()
        train_x, train_y = oversample.fit_resample(train_x, train_y)
        # summarize the new class distribution
        logger.info(f"counter after ADASYN is: {train_y[target].value_counts()}")
        return train_x, train_y
    except Exception as ex:
        logger.error(f"failed to run adaptive_synthetic_sampling_func due to: {ex}")

#####################################################ML model functions##########################################
def get_models_list_and_hyperparameters(seed_num):
    # TODO: still hardcoded the models and hyper parameters we used in the script
    # return back dictionary with the models list and the models hyperparameters
    models = {
        'LogisticRegression': LogisticRegression(solver='lbfgs', random_state=seed_num),
        'GaussianNB': GaussianNB(),
        # 'knn': KNeighborsClassifier(),
        'RandomForest': RandomForestClassifier(random_state=seed_num),
        # 'svm': SVC(probability=True, random_state=42)
        'GradientBoost': GradientBoostingClassifier(random_state=seed_num),
        'XGBoost': XGBClassifier()
    }

    LR_hyperparameters = {
        #'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    }

    GNB_hyperparameters = {}
    RF_hyperparameters = {
        'n_estimators': [10, 50, 100, 200, 400],
        'max_features': ['auto', 'sqrt', 0.33]
    }
    knn_hyperparameters = {
        'n_neighbors': [1, 2, 4, 5, 8, 10, 20, 50]
    }
    svm_hyperparameters = {
        'C': [0.1, 0.5, 1.0, 5.0, 10, 20, 40],
        'gamma': [0.005, 0.01, 0.05, 0.1, 1, 10],
    }
    GBM_hyperparameters = {#'n_estimators': [10, 20, 50, 100, 200, 400],
                           'n_estimators': [100, 200, 400, 500, 1500, 2000, 2500, 3000],
                           #'max_features': [2, 'auto', 'sqrt', 0.33],
                           #'learning_rate': [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
                           'learning_rate': [0.001, 0.01, 0.1]}
    XGB_hyperparameters = {}
    # Create hyper parameters dictionary
    hyperparameters = {
        'LogisticRegression': LR_hyperparameters,
        'GaussianNB': GNB_hyperparameters,
        #'knn': knn_hyperparameters,
        # 'svm': svm_hyperparameters,
        'RandomForest': RF_hyperparameters,
        'GradientBoost': GBM_hyperparameters,
        'XGBoost': XGB_hyperparameters
    }
    return models, hyperparameters


def best_estimator(X, y, kfold, scoring, model_name, model, hyperparameter):
    """
       Check for the optimal hyperparameters using Grid search
    """
    # Train the model
    clf = GridSearchCV(model, hyperparameter, scoring=scoring, cv=kfold, n_jobs=-1)
    clf.fit(X.values, y.values.ravel())
    #print(f"Best estimator for model {model_name} found by grid search:")
    #print(f"The best validation {scoring} is {clf.best_score_}")
    return clf.best_estimator_


def model_performance(model_name, model, kfold, train_x, train_y, test_x, test_y):
    clf = model
    logger.info("starting fit model with test data.")
    clf.fit(train_x.values, train_y.values.ravel())
    logger.info("calculate cross validation score over training.")
    # training dataset
    auc_score = cross_val_score(clf, train_x, train_y.values.ravel(), cv=kfold, scoring='roc_auc')
    acc_score = cross_val_score(clf, train_x, train_y.values.ravel(), cv=kfold, scoring='accuracy')
    rec_score = cross_val_score(clf, train_x, train_y.values.ravel(), cv=kfold, scoring='recall')
    precision_score = cross_val_score(clf, train_x, train_y.values.ravel(), cv=kfold, scoring='precision')
    #print("Metrics on Training dataset:")
    #print("             Mean,     Std")
    #print("----------------------------")
    #print(f"AUC:        {round(auc_score.mean(), 3), round(auc_score.std(), 3)}")
    #print(f"Accuracy:   {round(acc_score.mean(), 3), round(acc_score.std(), 3)}")
    #print(f"Recall:     {round(rec_score.mean(), 3), round(rec_score.std(), 3)}")
    #print(f"Precision:  {round(precision_score.mean(), 3), round(precision_score.std(), 3)}")
    logger.info("start predict prob' using test dataset.")
    if model_name == 'XGBoost':
        logger.info("for evaluate XGBoost will converted data into numpy ndarray ")
        train_x = train_x.to_numpy()
        test_x = test_x.to_numpy()
    # testing dataset
    y_pred = clf.predict(test_x)
    y_proba = clf.predict_proba(test_x)
    df_est_test = pd.DataFrame(y_proba)
    #fpr, tpr, thresholds = roc_curve(test_y, y_proba[:, 1])
    #roc_auc = round(auc(fpr, tpr), 3)
    logger.info("calculate ROC ..")
    fpr, tpr, th, roc_auc, optimalIndex = getRocData(test_y.values, y_proba[:, 1:])
    optimalTh = th[optimalIndex]
    logger.info("calculate confusion matrix.")
    est_y_binary = y_proba[:, 1] >= optimalTh
    cm = confusion_matrix(test_y.values[:, 0], est_y_binary)
    logger.info(f"Metrics on Testing dataset base on optimal Th: {cm}")
    tn, fp, fn, tp = confusion_matrix(test_y.values[:, 0], est_y_binary).ravel()
    logger.info(f"tn,fp,fn,tp {tn, fp, fn, tp}")
    accuracy = round((tn + tp) / (tp + tn + fp + fn), 3)
    sensitivity = round(tp / (tp + fn), 3)
    specificity = round(tn / (tn + fp), 3)
    ppv = round(tp/(tp+fp), 3)
    npv = round(tn/(tn+fn), 3)
    # adding 95% confidence interval AUC , df_est_test[1].values
    auc_ci = print_AUC_CI(test_y.values[:, 0], y_proba[:, 1])

    # logger.info("Metrics on Testing dataset:")
    # logger.info(f"{roc_auc} [AUC], {accuracy} [Accuracy], {sensitivity} [Sensitivity], {specificity} [Specificity]")
    return round(roc_auc, 3), auc_ci, accuracy, sensitivity, specificity, ppv, npv, df_est_test

def extract_feature_importance(model, train_x, train_y, target):
    model.fit(train_x, train_y)
    fi = pd.DataFrame({'feature': list(train_x.columns),
                       'importance': model.feature_importances_}). \
        sort_values('importance', ascending=False)
    fi['target'] = target
    return fi

def run_all_models(train_x, train_y, test_x, test_y, train, test, target, seed_num):
    # performance matrix
    ROC_AUC_list = []
    AUC_CI_list = []
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    ppv_list = []
    npv_list = []
    fi_RF = pd.DataFrame({})
    fi_GBM = pd.DataFrame({})
    fi_XGB = pd.DataFrame({})
    models, hyperparameters = get_models_list_and_hyperparameters(seed_num)
    for model_name in models:
        logger.info(f"---------start working on {model_name} model ----------")
        kfold = 3 # how many groups we want to split for the cross validation
        scoring = 'roc_auc'
        logger.info("start looking for best model ...")
        best_model = best_estimator(train_x, train_y, kfold, scoring, model_name, models[model_name], hyperparameters[model_name])
        # evaluate model performance
        logger.info("start evaluate best model performance ...")
        roc_auc, auc_ci, accuracy, sensitivity, specificity, ppv, npv, df_est_test = model_performance(model_name, best_model, kfold, train_x, train_y, test_x, test_y)
        test['est_pos' + model_name + '_' + target] = df_est_test[1].values
        if model_name == 'RandomForest':
            # save feature importance list
            fi_RF = extract_feature_importance(best_model, train_x, train_y, target)
        elif model_name == 'GradientBoost':
            fi_GBM = extract_feature_importance(best_model, train_x, train_y, target)
        elif model_name == 'XGBoost':
            fi_XGB = extract_feature_importance(best_model, train_x, train_y, target)
        ROC_AUC_list.append(roc_auc)
        AUC_CI_list.append(auc_ci)
        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)
        ppv_list.append(ppv)
        npv_list.append(npv)
    # # add AUC / Spec / Sens based on current ASCVD calculator
    # roc_auc, auc_ci, accuracy, sensitivity, specificity, npv, ppv = cur_calc_performance(test_y, test)
    # ROC_AUC_list.append(roc_auc)
    # AUC_CI_list.append(auc_ci)
    # accuracy_list.append(accuracy)
    # sensitivity_list.append(sensitivity)
    # specificity_list.append(specificity)
    # ppv_list.append(ppv)
    # npv_list.append(npv)
    model_list = [*models]
    # model_list.append('PCE')
    pd_performance = pd.DataFrame({
        'Model': model_list,
        'ROC AUC': ROC_AUC_list,
        'AUC_CI': AUC_CI_list,
        'Accuracy': accuracy_list,
        'Sensitivity': sensitivity_list,
        'Specificity': specificity_list,
        'PPV': ppv_list,
        'NPV': npv_list,
        'Target': target
    })

    pd_performance.sort_values(by='ROC AUC', ascending=False)
    return pd_performance, fi_RF, fi_GBM, fi_XGB, test


def getRocData(hot_y, y_score, wantedClass=0):
    i = wantedClass
    if ((hot_y.shape[1:] == ()) & (y_score.shape[1:] == ())):  # Vector ROC
        hot_y_use = hot_y
        y_score_use = y_score
    else:
        hot_y_use = hot_y[:, i]
        y_score_use = y_score[:, i]

    fpr, tpr, th = roc_curve(hot_y_use, y_score_use)
    roc_auc = auc(fpr, tpr)
    dist = np.sqrt((1 - tpr) ** 2 + (fpr) ** 2)
    optimalIndex = np.argmin(dist)
    return fpr, tpr, th, roc_auc, optimalIndex


def plotRoc(hot_y, y_score, wantedClass=0, color=None, legend='', linestyle='-', ax=None, fig=None, subTitle=''):
    if (fig != None):
        fig.show()
    # Compute ROC curve and ROC area for each class
    # Can also get "fig, (ax1, ax2,ax3) = plt.subplots(1, 3, sharey=True)"
    lw = 2
    fpr, tpr, th, roc_auc, optimalIndex = getRocData(hot_y, y_score, wantedClass)
    optimalTh = th[optimalIndex]
    if plt is None:
        logger.info('Cannot plot due to problem with matplotlib, Returned AUC and knee threshold')
    else:
        if (ax == None) | (fig == None):
            fig, (ax) = plt.subplots(1, 1, sharey=True)

        ax.plot(fpr, tpr, color=color,
                lw=lw, linestyle=linestyle, label=legend + ' (AUC = %0.3f)' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        ax.axis(xmin=0, xmax=1, ymin=0, ymax=1)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel('1 - Specificity', fontsize=12, fontname='DejaVu Sans')
        ax.set_ylabel('Sensitivity', fontsize=12, fontname='DejaVu Sans')
        ax.set_title(subTitle)
        ax.legend(loc="lower right", prop={'family': 'DejaVu Sans'})
        fig.suptitle('Receiver Operating Characteristic ', fontsize=20, fontweight='bold', fontname='DejaVu Sans')

    #  fig.canvas.draw()
    return roc_auc, optimalTh, fig, ax


def print_model_results(train_x,train_y,test_x,test_y,model):
    # Actual class predictions
    train_rf_predictions = model.predict(train_x)
    rf_predictions = model.predict(test_x)
    # Probabilities for each class
    train_rf_probs = model.predict_proba(train_x)[:, 1]
    rf_probs_train = model.predict_proba(train_x)
    rf_probs = model.predict_proba(test_x)
    # Calculate roc auc
    _,_,fig,ax = plotRoc(train_y.values,rf_probs_train[:,1:],legend='train')
    roc_auc,optimalTh,fig,ax = plotRoc(test_y.values,rf_probs[:,1:],legend='test',fig=fig,ax=ax)


######################################Main functions####################################################
def load_print_main_args(args):
    # load and print general variables
    data_f = args.data_f_name
    balance = args.balance
    seed_num = args.seed_num
    project_name = args.project_name
    over_sample = args.over_sample
    over_under_sample = args.over_under_sample
    borderline_smoth = args.borderline_smoth
    adaptive_synthetic_sampling = args.adaptive_synthetic_sampling

    logger.info(f"data file name is: {data_f}")
    logger.info(f"balance is: {balance}")
    logger.info(f"seed_num is: {seed_num}")
    logger.info(f"project_name is: {project_name}")
    logger.info(f"over_sample is: {over_sample}")
    logger.info(f"over_under_sample is: {over_under_sample}")
    logger.info(f"borderline_smoth is: {borderline_smoth}")
    logger.info(f"adaptive_synthetic_sampling is: {adaptive_synthetic_sampling}")

    return data_f, balance, seed_num, project_name, over_sample, over_under_sample, borderline_smoth, adaptive_synthetic_sampling

def load_print_config_args(config_path):
    # Parse configs from json
    with open(config_path, 'r') as j:
        config = json.load(j)
    path = config["save_path"]
    con_var_list = config["con_var_list"]
    bin_var_list = config["bin_var_list"]
    features = config["features"]
    targets = config["targets"]
    print("path is: ", path)
    print("con_var_list is: ", con_var_list)
    print("bin_var_list is: ", bin_var_list)
    print("features is: ", features)
    print("targets is: ", targets)

    return path, con_var_list, bin_var_list, features, targets


def drop_nulls(features_outcome_list, df):
    logger.info(f"number of patients before drop nulls: {len(df)}")
    df = df.dropna(subset=features_outcome_list)
    logger.info(f"number of patients after drop nulls: {len(df)}")
    return df

def drop_nulls_outcome(outcome, df):
    logger.info(f"number of patients before drop nulls is: {len(df)} for outcome: {outcome} ")
    df = df.dropna(subset=[outcome])
    logger.info(f"number of patients after drop nulls: {len(df)}")
    return df


def drop_nulls_one_by_one(features_list, df):
    logger.info(f"number of patients before drop nulls: {len(df)}")
    for feature in features_list:
        if feature.endswith('_N'):
            feature = feature[:-2]
        logger.info(f"null values in column {feature} is: {df[[feature]].isnull().sum().sum()}")
        df = df.dropna(subset=[feature])
        logger.info(f"number of patients after drop {feature} nulls: {len(df)}")
    return df


def convert_binary_vars_to_neg_one(df, var_list):
    # convert binary variables into 1/-1 instead
    for var in var_list:
        df[var] = np.select([df[var] == 1], [1], default=-1)
    return df


def prepare_dataset(df, features_list, bin_var_list):
    # converts empty cells into None in order to make sure "dropna" function will work
    df = df.replace(r'^\s*$', np.nan, regex=True)

    # drop nulls if found in one of the feature
    df = drop_nulls_one_by_one(features_list, df)

    # convert binary 0/1 to -1/1
    df = convert_binary_vars_to_neg_one(df, bin_var_list)
    return df


def split_dataset_pos_neg(df, target, seed_num):
    # divided into positive negative in order to make sure they are in the same balance between training / testing groups
    df_Endpoint_pos = shuffle(df[df[target] == 1])
    df_Endpoint_neg = shuffle(df[df[target] == 0])

    # for 2nd analysis we can reduce the number of negatives
    df_Endpoint_neg_1, df_Endpoint_neg_2 = split_group_by_prec(df=df_Endpoint_neg, prec_num=0.70, seed_num = seed_num)
    return df_Endpoint_pos, df_Endpoint_neg, df_Endpoint_neg_2


def print_and_save_results(pd_performance_all, fi_RF, fi_GBM, fi_XGB, path, time_f, name):
    pd_fi_RF = pd.DataFrame()
    pd_fi_GBM = pd.DataFrame()
    pd_fi_XGB = pd.DataFrame()
    pd_fi_RF = pd_fi_RF.append(fi_RF)
    pd_fi_GBM = pd_fi_GBM.append(fi_GBM)
    pd_fi_XGB = pd_fi_XGB.append(fi_XGB)
    logger.info("####################################Final-results############################################")
    logger.info(pd_performance_all)
    pd_performance_AUC_pivot = pd.pivot_table(pd_performance_all[['Model', 'ROC AUC', 'Target']], values='ROC AUC',
                                              index=['Model'],
                                              columns=['Target'], aggfunc=np.sum)
    pd_performance_all.to_csv(path+name+"_final_full_results_"+time_f+"_.csv", index=False)
    logger.info("column names of pd_performance_AUC_pivot:")
    pd_performance_AUC_pivot.columns
    pd_performance_AUC_pivot.sort_values(by=pd_performance_AUC_pivot.columns[0], ascending=False)
    logger.info("####################################Final-pivot-results############################################")
    logger.info(pd_performance_AUC_pivot)
    pd_performance_AUC_pivot.to_csv(path+name+"_final_AUC_results_"+time_f+".csv")
    logger.info("###############################Random Forest- feature importance####################################")
    logger.info(pd_fi_RF)
    pd_fi_RF.to_csv(path + name + "_RF_final_feature_importance_results_" + time_f + ".csv", index=False)
    logger.info("###############################GBM- feature importance####################################")
    logger.info(pd_fi_GBM)
    pd_fi_GBM.to_csv(path + name + "_GBM_final_feature_importance_results_" + time_f + ".csv", index=False)
    logger.info("###############################XGB- feature importance####################################")
    logger.info(pd_fi_XGB)
    pd_fi_XGB.to_csv(path + name + "_XGB_final_feature_importance_results_" + time_f + ".csv", index=False)


#########################################MAIN###########################################################
argpar = ArgumentParser()
# Path of config file
argpar.add_argument("-cp", "--config_path", required=True)
argpar.add_argument("-df", "--data_f_name", required=True)
#argpar.add_argument("-tl", "--target_list", nargs='+',  required=True)
argpar.add_argument("-b", "--balance", default=False)
argpar.add_argument("-sn", "--seed_num", type=int, default=0)
argpar.add_argument("-pn", "--project_name", default="")
argpar.add_argument("-os", "--over_sample", default=False)
argpar.add_argument("-ous", "--over_under_sample", default=False)
argpar.add_argument("-bs", "--borderline_smoth", default=False)
argpar.add_argument("-adasym", "--adaptive_synthetic_sampling", default=False)

args = argpar.parse_args()
config_path = args.config_path
# extract and print argument from json configuration file
path, con_var_list, bin_var_list, features, targets = load_print_config_args(config_path)

#-----------------Log-config------------------
time_f = datetime.now().strftime('%m%d%Y_%H%M')
project_name = args.project_name
log = path + project_name +"_classifier_logger_" + time_f + ".log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    handlers=[logging.FileHandler("{0}".format(log)), logging.StreamHandler()])
logger = logging.getLogger('main')

if wandb_enable:
    import wandb
    #--------------- Wandb configuration ----------------#
    outcomes = args.target_list
    dataDir = save_path
    wandb_runname = 'tabular data - features 1'
    if len(outcomes) > 1:
        outcome = 'multiple'
    else:
        outcome = outcomes[0]
    noWandbLog = 0
    erasePrevious = 1  # erase a previous run by the same name?  only set to 1 if debugging
    runGroup = "Paper1"
    modelName = "traditional ML"  # "basicConvNet"  #choices:  basicConvNet smallerVGGNet abtfInceptionV3
    runNotes = 'tabular data'
    runTags = ['paper1', 'CED-Detection', 'features1', 'Win', str(outcome)]

    projectName = 'tabular'

    runName = wandb_runname + "_" + modelName

    # this needs to happen before wandb.init
    if noWandbLog:
        os.environ['WANDB_MODE'] = 'dryrun'

    wandb.init(project=projectName, name=runName, notes=runNotes, dir=dataDir, tags=runTags)

    config_wandb = wandb.config
    config_wandb.folder = runGroup
    config_wandb.batch_size = batch_size
    config_wandb.initialLearningRate = '1e-3'
#---------------------------------------------#

def main(args, features, path, con_var_list, bin_var_list, targets):
    subset = False
    logger.info(f"tabular data classifier script starts at: {datetime.now().strftime('%m%d%Y_%H%M')}")
    # general variables:
    data_f_name, balance, seed_num, name, over_sample, over_under_sample, borderline_smoth, adaptive_synthetic_sampling = load_print_main_args(args)
    time_f = datetime.now().strftime('%m%d%Y_%H%M')

    # load the data
    df_row = data_load(path+data_f_name)
    #------- for creating learning curve - select subset of the dataset----------
    if subset:
        y_mrn = df_row[['MRN', targets[0]]].drop_duplicates()

        subset1_mrn, subset2_mrn = train_test_split(y_mrn, train_size=0.8,
                                                 stratify=y_mrn[targets[0]],
                                                 random_state=seed_num)
        logger.info(f"dataset size before split is: {len(df_row)}")
        df_row = df_row.merge(subset1_mrn[['MRN']], on=['MRN'])
        logger.info(f"dataset size after split is: {len(df_row)}")
    #----------------------------------------------------------------------------
    # prepare dataset
    df = prepare_dataset(df_row, features, bin_var_list)
    logger.info("update df columns is:")
    logger.info(df.columns)
    # save dataset
    df.to_csv(path+name+'_cohort_'+time_f+'.csv', index=False)
    pd_performance_all = pd.DataFrame()
    fi_RF_all = pd.DataFrame()
    fi_GBM_all = pd.DataFrame()
    fi_XGB_all = pd.DataFrame()
    for target in targets:
        logger.info(f"################################{target}-Analysis########################################")
        # drop nulls for outcome
        df = drop_nulls_outcome(target, df)
        # divided dataset to pos/neg
        df_pos, df_neg, df_subset_neg = split_dataset_pos_neg(df, target, seed_num)

        # divided dataset to final training test groups
        if balance is False:
            logger.info("using imbalance data")
            train_x, train_y, test_x, test_y, train, test = create_train_test(df_pos, df_neg, features, target, time_f, path, con_var_list, seed_num, name)
        else:
            logger.info("using balance data")
            train_x, train_y, test_x, test_y, train, test = create_train_test(df_pos, df_subset_neg, features, target, time_f, path, con_var_list, seed_num, name)
        # add over sample
        if over_sample:
            train_x, train_y = over_sample_func(train_x, train_y, target)
        if over_under_sample:
            train_x, train_y = over_under_sample_func(train_x, train_y, target)
        if borderline_smoth:
            train_x, train_y = borderline_smoth_func(train_x, train_y, target)
        if adaptive_synthetic_sampling:
            train_x, train_y = adaptive_synthetic_sampling_func(train_x, train_y, target)
        # using all models to train and test
        pd_performance, fi_RF, fi_GBM, fi_XGB, test_all_est = run_all_models(train_x, train_y, test_x, test_y, train, test, target, seed_num)
        logger.info(pd_performance)
        pd_performance_all = pd_performance_all.append(pd_performance)
        logger.info(f"will save test prob' results for target: {target}")
        test_all_est.to_csv(path+'test_all_models_est_results_'+target+'_'+time_f+'.csv', index=False)
        fi_RF_all = fi_RF_all.append(fi_RF)
        fi_GBM_all = fi_GBM_all.append(fi_GBM)
        fi_XGB_all = fi_XGB_all.append(fi_XGB)
    print_and_save_results(pd_performance_all, fi_RF_all, fi_GBM_all, fi_XGB_all, path, time_f, name)
    logger.info(f"tabular data classifier script ends at: {datetime.now().strftime('%m%d%Y_%H%M')}")


main(args, features, path, con_var_list, bin_var_list, targets)
