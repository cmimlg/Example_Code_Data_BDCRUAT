import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from math import ceil, floor, exp, sqrt, log
import sys, traceback
import GPy
from GPy.kern import *
import random
import climin
import sys
import time
import os
import matplotlib.pyplot as plt
import logging
import csv
from memory_profiler import memory_usage
import statsmodels.api as sm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.metrics import accuracy_score
import warnings
from sklearn import linear_model, neighbors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

model_dict = dict()
scoring_dict = dict()
trng_clusters = None
test_clusters = None
ts_cluster_acc = None
trng_cluster_acc = None
ts_bdt_cluster_acc = None
trng_bdt_cluster_acc = None

# create logger for the application
logger = logging.getLogger('Airline Delay Logger')

ch = logging.StreamHandler()



df_adc_trng = None
df_adc_test = None
AD_COMPLETED = False
READ_COMPLETED = False
df_trng = None
df_test = None




# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)


logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
DATA_READ_COMPLETED = False
AD_COMPLETED = False

def read_data(test_set_prop = 0.3, N_prop = 1.0):
    global df_trng, df_test, DATA_READ_COMPLETED
    if not DATA_READ_COMPLETED:
        os.chdir("/home/admin123/bdcruat/regression")
        fp = "jan_feb_pp_data.csv"
        df = pd.read_csv(fp)       
        N = len(df.index)
        N_exp = int(N*N_prop) - 1
        df = df.sample(frac = N_prop, random_state = 1254)
        test_set_size = int(ceil(N_exp * test_set_prop))
        random.seed(a=42)
        test_rows = random.sample(df.index, test_set_size)
        df_trng = df.drop(test_rows)
        df_test = df.ix[test_rows]
        DATA_READ_COMPLETED = True


    return df_trng, df_test

def read_data2():
    global df_trng, df_test, DATA_READ_COMPLETED
    if not DATA_READ_COMPLETED:
        os.chdir("/home/admin123/bdcruat/regression")
        fp_trng = "cal_housing_trng_filtered.csv"
        fp_test = "cal_housing_test_filtered.csv"
        df_trng = pd.read_csv(fp_trng)
        df_test = pd.read_csv(fp_test)
        DATA_READ_COMPLETED = True
    return df_trng, df_test

def run_base_DT(leaf_size = 70, doAnomalyDetection = True):
    if doAnomalyDetection:
        df_trng, df_test = do_anomaly_detection()
    else:
        df_trng, df_test = read_data()
    
    preds = df_trng.columns.tolist()
    
    preds.remove('ARR_DELAY')
    if "leaf_id" in preds:
        df_trng.drop("leaf_id", axis = 1, inplace = True)
        df_test.drop("leaf_id", axis = 1, inplace = True)
        preds.remove("leaf_id")
    
    X = df_trng[preds].as_matrix()
    Y = df_trng['ARR_DELAY'].as_matrix()
    Xt = df_test[preds].as_matrix()
    Yt = df_test['ARR_DELAY'].as_matrix()

    reg =  DecisionTreeRegressor(min_samples_leaf = leaf_size,\
                                 random_state = 1254)
    reg.fit(X,Y)
    Ytest_pred = reg.predict(Xt)
    
    score_test = calc_rmse(Yt, Ytest_pred)
    Ytrng_pred = reg.predict(X)
    
    score_trng = calc_rmse(Y, Ytrng_pred)
    
    logger.info("Regression Tree RMSE - training: " + str(score_trng))
  
    logger.info("Regression Tree RMSE - test: " + str(score_test))
    

    return score_trng, score_test

def score_base_dt_test(test_clusters, reg):
    global ts_bdt_cluster_acc

    ts_bdt_cluster_acc = dict()

    for l in test_clusters:
        df_seg = test_clusters[l]
        preds = df_seg.columns.tolist()
        preds.remove('ARR_DELAY')
        preds.remove('leaf_id')
        X = df_seg[preds].as_matrix()
        Y = df_seg["ARR_DELAY"].as_matrix()
        lsd = scoring_dict[l]
        md = model_dict[l]
        Ypl = reg.predict(X)
        reg_accuracy = calc_rmse(Y, Ypl)
        ts_bdt_cluster_acc[l] = reg_accuracy
    
    return

    

def calc_rmse(ytrue, ypred):
    N = ytrue.shape[0]
    err = ypred - ytrue
    se = err * err
    rmse = sqrt(sum(se)/float(N))
    return rmse

def cluster_tree(leaf_size = 100,\
                 drawTree = False,\
                 retAcc = False,
                 doAnomalyDetection = True):

    global model_dict, trng_clusters, test_clusters, num_cols, scoring_dict
    model_dict.clear()
    scoring_dict.clear()

    if doAnomalyDetection:
        df_trng, df_test = do_anomaly_detection()
    else:
        df_trng, df_test = read_data()
    
    preds = df_trng.columns.tolist()
    cnames = df_trng["ARR_DELAY"].unique()
    ccnames = [str(c) for c in cnames]
    
    preds.remove('ARR_DELAY')
    if "leaf_id" in preds:
        df_trng.drop("leaf_id", axis = 1, inplace = True)
        df_test.drop("leaf_id", axis = 1, inplace = True)
        preds.remove("leaf_id")
    
    X = df_trng[preds].as_matrix()
    Y = df_trng['ARR_DELAY'].as_matrix()
    Xt = df_test[preds].as_matrix()
    Yt = df_test['ARR_DELAY'].as_matrix()

    
    reg =  DecisionTreeRegressor(min_samples_leaf = leaf_size,\
                                 random_state = 1254)
    reg.fit(X,Y)
    Ytest_pred = reg.predict(Xt)
   
    score_test = calc_rmse(Yt, Ytest_pred)
    Ytrng_pred = reg.predict(X)
    
    score_trng = calc_rmse(Y, Ytrng_pred)
    
    logger.info("Decision Tree Score Training: " + str(score_trng))
  
    logger.info("Decision Tree Score Test: " + str(score_test))
    
    Xl = reg.apply(X)
    logger.debug("Starting decision tree based segmentation for leaf size: " +\
                 str(leaf_size))
    num_clusters = len(np.unique(Xl))
    logger.debug("Segmentation created " + str(num_clusters) + " clusters!")

    utcl = np.unique(Xl)
    df_trng["leaf_id"] = Xl.reshape(Xl.shape[0],1)
    trng_clusters = dict()
    for lid in utcl:
        df_seg = df_trng[df_trng.leaf_id == lid]
        preds = df_seg.columns.tolist()
        preds.remove("leaf_id")
        preds.remove("ARR_DELAY")
        Xs = df_seg[preds].as_matrix()
        Ys = df_seg["ARR_DELAY"].as_matrix()
        Yps = reg.predict(Xs)
        acc_leaf_trng = calc_rmse(Ys, Yps)
        trng_clusters[lid] = df_seg
        model_dict[lid] = dict()
        model_dict[lid]["base_leaf"] = reg
        scoring_dict[lid] = dict()
        scoring_dict[lid]["base_leaf"] = acc_leaf_trng
    test_clusters = dict()
    Xtl = reg.apply(Xt)
    df_test["leaf_id"] = Xtl.reshape(Xtl.shape[0],1)
    utstcl = np.unique(Xtl)
    for lid in utstcl:
        df_seg = df_test[df_test.leaf_id == lid]
        test_clusters[lid] = df_seg
    
    if drawTree:
        tree.export_graphviz(clf, filled = True, feature_names = preds,\
                             class_names = ccnames,\
                             out_file='fc_dtree.dot')
        logger.debug("Tree Visual exported!")
 
    logger.debug("Done with regression tree segmentation.... ")

    score_base_dt_test(test_clusters, reg) 
    
    
    if not retAcc:
        return 
    else:
        return score_trng, score_test



def fit_leaf_gp():
    global model_dict, test_clusters, trng_clusters, num_cols, scoring_dict

    if not trng_clusters:
        logger.debug("Need to run decision tree segmentation prior")
        return
 
    
    start_time = time.time()
    for l in trng_clusters:
        logger.debug("Processing cluster : " + str(l))
        df_seg = trng_clusters[l]
        df_seg = df_seg.reset_index(drop = True)
        preds = df_seg.columns.tolist()

        preds.remove('ARR_DELAY')
        preds.remove('leaf_id')
        N = int(df_seg.shape[0])
        Ntrng = int(N * 0.667)
        Xtrng = df_seg.ix[:Ntrng, preds].as_matrix()
        Ytrng = df_seg.ix[:Ntrng, "ARR_DELAY"].as_matrix()
        Xt = df_seg.ix[(Ntrng + 1):, preds].as_matrix()
        Yt = df_seg.ix[(Ntrng + 1):, "ARR_DELAY"].as_matrix()
        Yt = Yt.reshape((Yt.shape[0], 1))
        Ytrng = Ytrng.reshape((Ytrng.shape[0], 1))
        k = Bias(input_dim = 8) +\
            Poly (input_dim = 8, order = 2) + \
            RBF(input_dim = 8, ARD = True)
            
        
        m  = GPy.models.GPRegression(Xtrng, Ytrng, k)
        m.optimize()
        
        
        Ytest_pred = m.predict(Xt)[0]
        Ytest_pred = Ytest_pred.ravel()
        Yt = Yt.ravel()
        acc_leaf_trng_gp = calc_rmse(Yt, Ytest_pred)
        model_dict[l]["gp"] = m
        scoring_dict[l]["gp"] = acc_leaf_trng_gp

    
    return

def scorer(m, X, Y):
    Yp = m.predict(X)
    rmse = calc_rmse(Y, Yp)
    return rmse

def fit_leaf_lr():
    global model_dict, test_clusters, trng_clusters, num_cols, scoring_dict

    if not trng_clusters:
        logger.debug("Need to run decision tree segmentation prior")
        return
 
    
    start_time = time.time()
    for l in trng_clusters:
        logger.debug("Processing cluster : " + str(l))
        df_seg = trng_clusters[l]
        preds = df_seg.columns.tolist()
        
        preds.remove('ARR_DELAY')
        preds.remove('leaf_id')

        X = df_seg.ix[:, preds].as_matrix()
        Y = df_seg.ix[:, "ARR_DELAY"].as_matrix()

        m = linear_model.LinearRegression(normalize = True)
        m.fit(X,Y)
        acc_leaf_trng_lr = np.mean(cross_val_score(m, X, Y,\
                                                   scoring = scorer, cv = 3))
        logger.debug("RMSE Linear Regression: " + str(acc_leaf_trng_lr))
        model_dict[l]["lr"] = m
        scoring_dict[l]["lr"] = acc_leaf_trng_lr

    
    return



def run_algorithms(leaf_size = 1000, score_trng = False):
    nn = 3
    draw_Tree = False
    retAcc = False
    anomalyDetection = True
    logger.info("Running Decision Tree Segmentation...")
    cluster_tree(leaf_size, draw_Tree, retAcc, anomalyDetection)
    logger.info("Done with decison tree segmentation!")
    logger.info("Running GP Regression on Segments...")
    fit_leaf_gp()
    logger.info("Done with leaf GP regression!")
    logger.info("Running LR on Segments...")
    fit_leaf_lr()
    logger.info("Done with LR!")


    logger.info("Doing the scoring...")
    acc_test, tc = score_test_set()
    
    logger.info("Done!")
    if score_trng:
        acc_trng = score_training_set()
        return acc_trng, acc_test
    else:
        return tc

def filter_noise():
    global test_clusters, trng_clusters
    logger.debug("Starting the filtering analysis... ")
    tc = run_algorithms(70, False)
    seg_errs = {id: min(tc[id].values()) for id in tc}
    threshold = np.percentile(seg_errs.values(), 90)
    bpc = { id: tc[id] for id in tc if min(tc[id].values()) >= threshold}
    
    os.chdir("/home/admin123/bdcruat/regression")
    trng_bpc_data = pd.concat([trng_clusters[lid] for lid in bpc],\
                              axis = 0)
    trng_bpc_data.reset_index(drop = True, inplace = True)
    cols = trng_bpc_data.columns.tolist()
    cols.remove("leaf_id")
    trng_bpc_data = trng_bpc_data[cols]
    fp_trng_ol = "cal_housing_ol_trng.csv"
    trng_bpc_data.to_csv(fp_trng_ol, index = False)
    test_bpc_data = pd.concat([test_clusters[lid] for lid in bpc],\
                              axis = 0)
    test_bpc_data.reset_index(drop = True, inplace = True)
    test_bpc_data = test_bpc_data[cols]
    fp_test_ol = "cal_housing_ol_test.csv"
    test_bpc_data.to_csv(fp_test_ol, index = False)
    
    for seg in bpc:
        del trng_clusters[seg]
        del test_clusters[seg]
    logger.info("Rescoring the test set....")
    ts = score_test_set()

    trng_good_seg_df = pd.concat([trng_clusters[lid] for lid in trng_clusters],\
                              axis = 0)
    trng_good_seg_df.reset_index(drop = True, inplace = True)
    trng_good_seg_df = trng_good_seg_df[cols]
    test_good_seg_df = pd.concat([test_clusters[lid] for lid in test_clusters],\
                              axis = 0)
    test_good_seg_df.reset_index(drop = True, inplace = True)
    test_good_seg_df = test_good_seg_df[cols]
    fp_trng_good = "cal_housing_trng_filtered.csv"
    trng_good_seg_df.to_csv(fp_trng_good, index = False)
    fp_test_good = "cal_housing_test_filtered.csv"
    test_good_seg_df.to_csv(fp_test_good, index = False)


    return 

def score_test_set():
    global model_dict, scoring_dict, test_clusters, ts_cluster_acc
    Yt = list()
    Yp = list()
    ts_cluster_acc = dict()
    
    for l in test_clusters:
        df_seg = test_clusters[l]
        preds = df_seg.columns.tolist()
        preds.remove('ARR_DELAY')
        preds.remove('leaf_id')
        X = df_seg[preds].as_matrix()
        Y = df_seg["ARR_DELAY"].as_matrix()
        lsd = scoring_dict[l]
        md = model_dict[l]
        base_leaf_score = lsd["base_leaf"]
        lr_leaf_score = lsd["lr"]
        gp_score = lsd["gp"]
        #knn_score = lsd["knn"]
        csdict = {"bl": base_leaf_score, "lr": lr_leaf_score, \
                  "gp":gp_score}
        score_array = np.array([base_leaf_score, lr_leaf_score,\
                                gp_score])
        est_array = np.array([md["base_leaf"], md["lr"], \
                              md["gp"]])
        min_index = np.argmin(score_array)

        best_est = est_array[min_index]
        Yt.extend(Y)
        if min_index == 2:
            Ypl = best_est.predict(X)[0]
        else:
            Ypl = best_est.predict(X)
        Ypl = Ypl.ravel()
        Yp.extend(Ypl)
        est_accuracy = calc_rmse(Y, Ypl)
        ts_cluster_acc[l] = est_accuracy
    Yta = np.array(Yt)
    Ypa = np.array(Yp)
    acc_test = calc_rmse(Yta, Ypa)

    logger.debug("Accuracy for test set is : " + str(acc_test))
        
        
    return acc_test, ts_cluster_acc

def score_training_set():
    global model_dict, scoring_dict, trng_clusters
    Yt = list()
    Yp = list()
    ts_cluster_acc = dict()
    for l in trng_clusters:
        df_seg = trng_clusters[l]
        preds = df_seg.columns.tolist()
        preds.remove('ARR_DELAY')
        preds.remove('leaf_id')
        X = df_seg[preds].as_matrix()
        Y = df_seg["ARR_DELAY"].as_matrix()
        lsd = scoring_dict[l]
        md = model_dict[l]
        base_leaf_score = lsd["base_leaf"]
        lr_leaf_score = lsd["lr"]
        gp_score = lsd["gp"]
        #knn_score = lsd["knn"]
        csdict = {"bl": base_leaf_score, "lr": lr_leaf_score, \
                  "gp":gp_score}
        score_array = np.array([base_leaf_score, lr_leaf_score,\
                                gp_score])
        est_array = np.array([md["base_leaf"], md["lr"], \
                              md["gp"]])
        min_index = np.argmin(score_array)

        best_est = est_array[min_index]
        Yt.extend(Y)
        if min_index == 2:
            Ypl = best_est.predict(X)[0]
        else:
            Ypl = best_est.predict(X)
        Ypl = Ypl.ravel()
        Yp.extend(Ypl)
        est_accuracy = calc_rmse(Y, Ypl)
        ts_cluster_acc[l] = est_accuracy
    Yta = np.array(Yt)
    Ypa = np.array(Yp)
    acc_trng = calc_rmse(Yta, Ypa)
    logger.debug("Accuracy for training set is : " + str(acc_trng))
        
        
    return acc_trng




def exp_leaf_size_effect():
    leaf_sizes = np.arange(400, 1400, 200)
    test_accuracy = np.zeros(len(leaf_sizes))
    trng_accuracy = np.zeros(len(leaf_sizes))
    score_trng = True
    
    for v, ls in enumerate(leaf_sizes):
        logger.debug("Running exp leaf size: " + str(ls))
        acc_trng, acc_test = run_algorithms(ls,score_trng) 
        test_accuracy[v] = acc_test
        trng_accuracy[v] = acc_trng
        
    axes = plt.gca()
    axes.set_xlim([300, 1500])
    axes.set_ylim([7.5, 9.0])
    plt.figure(1)
    plt.scatter(leaf_sizes, test_accuracy, color = "red", label = "Test Accuracy")
    plt.scatter(leaf_sizes, trng_accuracy, color = "blue", label = "Training Accuracy")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Leaf Size Versus Training and Test RMSE - Complete Model")
    plt.grid()
    plt.legend(loc = 2)
    plt.show()

    return 

def exp_leaf_size_reg_tree_effect():
    leaf_sizes = np.arange(200, 1200, 200)
    test_err = np.zeros(len(leaf_sizes))
    trng_err = np.zeros(len(leaf_sizes))    
    anomalyDetection = True
    for v, ls in enumerate(leaf_sizes):
        logger.debug("Running exp leaf size: " + str(ls))
        score_trng, score_test = run_base_DT(ls, True)
        test_err[v] = score_test
        trng_err[v] = score_trng
        
    plt.scatter(leaf_sizes, test_err, color = "red", label = "Test RMSE")
    plt.scatter(leaf_sizes, trng_err, color = "blue", label = "Training RMSE")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.title("Leaf Size Versus Training and Test RMSE - Regression Tree")
    plt.grid()
    plt.legend(loc = 4)
    plt.show()

    return


def segment_profile_plot():
    global test_clusters
    seg_ids = test_clusters.keys()
    num_clusters = len(seg_ids)
    seg_mean_vals = np.zeros(num_clusters)
    for v,c in enumerate(test_clusters):
        seg_mean_vals[v] = test_clusters[c]["ARR_DELAY"].mean()
    plt.scatter(seg_ids, seg_mean_vals, color = "blue")
    plt.title("ARR_DELAY (mean) by Segment ID")
    plt.ylabel("ARR_DELAY (mean) for Segment")
    plt.xlabel("Segment ID")
    plt.grid()
    plt.legend(loc = "lower right")
    plt.show()  
    
    return



def do_anomaly_detection():

    global df_adc_trng, df_adc_test, AD_COMPLETED
    logger.debug("Doing Anomaly Detection ... ")
  
    if not AD_COMPLETED:
        col_names = ['MONTH','DAY_OF_MONTH', 'DAY_OF_WEEK', 'DEP_DELAY', 'TAXI_OUT',\
             'TAXI_IN', 'CRS_ELAPSED_TIME', 'NDDT', 'NDAT','ARR_DELAY']
        
        rng = np.random.RandomState(42)
        df_trng, df_test = read_data()
        X = df_trng.ix[:, 0:9].as_matrix()
        Y = df_trng.ix[:, 9].as_matrix()
        Xt = df_test.ix[:, 0:9].as_matrix()
        Yt = df_test.ix[:, 9].as_matrix()
        
     

        clf = IsolationForest(max_samples=500, random_state=rng, contamination = 0.25)
        clf.fit(X)
        trng_outliers = clf.predict(X)
        test_outliers = clf.predict(Xt)

        Y = Y.reshape((Y.shape[0],1))
        Yt = Yt.reshape((Yt.shape[0],1))
        df_adc_trng = pd.DataFrame(np.hstack((X,Y)))
        df_adc_test = pd.DataFrame(np.hstack((Xt,Yt)))
        df_adc_trng.columns = col_names
        df_adc_test.columns = col_names

        df_adc_trng["outlier"] = trng_outliers
        df_adc_test["outlier"] = test_outliers


        df_adc_trng = df_adc_trng[df_adc_trng["outlier"] == 1]
        df_adc_trng = df_adc_trng.drop("outlier", 1)

        df_adc_test = df_adc_test[df_adc_test["outlier"] == 1]
        df_adc_test = df_adc_test.drop("outlier", 1)
        AD_COMPLETED = True


 

    logger.debug("Anomaly Detection Completed !")
    return df_adc_trng, df_adc_test

def fit_xgb(anomaly_detection = False):
    el = list()
    if anomaly_detection:
        df_trng, df_test = do_anomaly_detection()
        X = df_trng.ix[:, 0:9].as_matrix()
        Y = df_trng.ix[:, 9].as_matrix()
        Xt = df_test.ix[:, 0:9].as_matrix()
        Yt = df_test.ix[:, 9].as_matrix()

    else:
        df_trng, df_test = read_data()
        X = df_trng.ix[:, 0:9].as_matrix()
        Y = df_trng.ix[:, 9].as_matrix()
        Xt = df_test.ix[:, 0:9].as_matrix()
        Yt = df_test.ix[:, 9].as_matrix()
        
    dtrain = xgb.DMatrix(X, label = Y)
    dtest = xgb.DMatrix(Xt, label = Yt)

    num_rounds = 200
    param = {'max_depth':4, 'eta':0.5, \
             'silent':1, 'subsample': 0.5 ,\
             'colsample_bytree': 0.5 }
    bst = xgb.train(param, dtrain, num_rounds)
    
    Ytp = bst.predict(dtest)
    err = Yt - Ytp
    el.extend(err)
    ea = np.array(el)
    se = ea * ea
    rmse = sqrt(sum(se)/Xt.shape[0])
    logger.debug("Size of test set is :  " + str(len(el)))
    logger.debug("RMSE for test set is: " + str(rmse))


    return

def fit_rf(anomaly_detection = False):
    el = list()
    if anomaly_detection:
        df_trng, df_test = do_anomaly_detection()
    else:
        df_trng, df_test = read_data()
    preds = df_trng.columns.tolist()
    preds.remove("ARR_DELAY")
    X = df_trng[preds].as_matrix()
    Y = df_trng["ARR_DELAY"].as_matrix()
    Xt = df_test[preds].as_matrix()
    Yt = df_test["ARR_DELAY"].as_matrix()

    reg =  RandomForestRegressor(n_estimators=200)
    
    reg = reg.fit(X,Y)
    Ytp = reg.predict(Xt)
    err = Yt - Ytp
    el.extend(err)
    ea = np.array(el)
    se = ea * ea
    rmse = sqrt(sum(se)/Xt.shape[0])
    logger.debug("Size of test set is :  " + str(Yt.shape[0]))
    logger.debug("Accuracy for test set is: " + str(rmse))



    return

def segment_LRE_plot():
    global ts_cluster_acc, ts_bdt_cluster_acc
    acc_trng, acc_test = run_algorithms(1000, True)
    seg_ids = ts_cluster_acc.keys()
    bdt_vals = ts_bdt_cluster_acc.values()
    dtc_vals = ts_cluster_acc.values()
    plt.scatter(seg_ids, bdt_vals, color = "red", label = "CART RT")
    plt.scatter(seg_ids, dtc_vals,\
                color = "blue", label = "Augmented RT")
    plt.title("Test Set RMSE By Segment ID")
    plt.ylabel("RMSE")
    plt.xlabel("Segment ID")
    plt.grid()
    plt.legend(loc = "lower right")
    plt.show()   
    return

    
    
