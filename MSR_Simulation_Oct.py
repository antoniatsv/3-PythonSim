import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import statistics
from torch import relu, tanh, sigmoid
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from itertools import product
from time import time
import os

def generate_x_fnc() -> np.ndarray:
    X_1 = np.random.binomial(n = 1, p = 0.5, size = 1)
    X_2 = np.random.normal(loc = 1, size=1) #loc (mean) should be higher because we want X to be distributed around positive numbers
    
    return X_1, X_2

def generate_u_fnc(u_dim:int) -> np.ndarray:
    U = np.random.normal(size = (u_dim))
    return U

def specify_gammas_fnc() -> Tuple[np.ndarray]:
    gamma0  = gamma0
    gamma_X1 = gamma_X1
    gamma_X2 = gamma_X1
    gamma_U = gamma_X1
    return gamma_X1, gamma_X2, gamma_U, gamma0

def generate_y_fnc(X_1, X_2, U, gamma_X1, gamma_X2, gamma_U, gamma0) -> int:
    p_y = sigmoid(torch.tensor(gamma0 + gamma_X1 * X_1 + gamma_X2 * X_2 + gamma_U * U)).numpy()
    assert 0 < p_y < 1
    Y = np.random.binomial(n=1, p=p_y)
    return Y

def generate_z_fnc(X_1, X_2, W, b, stddev, nonlinearity: str = 'relu') -> np.ndarray:
    
    X = np.concatenate([X_1, X_2])

    in_dim = X.shape[0]
    out_dim = W.shape[0]
    
    assert W.shape == (out_dim, in_dim)
    assert b.shape[0] == out_dim
    
    pre_nonlinearity = np.matmul(W, X) + b

    if nonlinearity == 'relu':
        Z_mean = relu(torch.tensor(pre_nonlinearity)).numpy()
    elif nonlinearity == 'tanh':
        Z_mean = tanh(torch.tensor(pre_nonlinearity)).numpy()
    else:
        raise NotImplementedError(f'Requested nonlinearity {nonlinearity} is not implemented yet.')

    Z = np.random.normal(loc=Z_mean, scale=stddev)
    assert Z.shape[0] == out_dim

    return Z

def specify_W_and_b_fnc(x_dim: int, z_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    
     assert x_dim > 1
    
     W11 = W[0, 0]
     W21 = W[1, 0]
     W31 = W[2, 0]
     W12 = W[0, 1]
     W22 = W[1, 1]
     W32 = W[2, 1]
    
    
     W = np.array([
      [W11, W12],
      [W21, W22],
      [W31, W32], 
    ])
    
    
     b1 = b1
     b2 = b2
     b3 = b3
    
     b = np.array([b1, b2, b3])
    
     W = (z_dim, x_dim)
     b = (z_dim)
   
    
     return W, b

def generate_test_data_fnc(N_test,  
                           W,
                           b,
                           gamma0,
                           gamma_X1,
                           gamma_X2,
                           gamma_U,
                           stddev) -> None:
       
    x_dim = 2
    z_dim = 3
    u_dim = 1
    
    column_headers = [f'X_{i}' for i in range(x_dim)] + [f'Z_{i}' for i in range(z_dim)] + ['U', 'Y']
    print(f'Column names will be {column_headers}')
    #output_file = open(output_path, 'w')
   
    #output_file.write(','.join(column_headers) + '\n')
    return_array = []
    
   
    
    for n in range(N_test):
           X = generate_x_fnc()
           U = generate_u_fnc(u_dim)
           Z = generate_z_fnc(X_1 = X[0], X_2 = X[1], W = W, b = b, nonlinearity = 'relu', stddev = stddev)
           Y = generate_y_fnc(X_1 = X[0], X_2 = X[1], U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)
           
        # Concatenate these together
           output_line_test = np.concatenate([X[0], X[1], U, Z, Y]).tolist()
           return_array.append(output_line_test)
        # Turn them into string-values
           output_line_test = [str(val) for val in output_line_test]
        # Combine them into a comma-separated llist
           output_line_test = ','.join(output_line_test)
        # Write to file and add newline
           #output_file.write(output_line + '\n')
        
    #output_file.close()
    #print(f'Wrote {N_test} lines to {output_path}')
    
    return np.array(return_array)
 

def generate_R1_fnc(R1_prev, Z_11, Z_12, Z_22, U, beta0, beta_X1, beta_X2, beta_U) -> Tuple[np.ndarray]:
    
    Z_11 = Z_11
    Z_12 = Z_12
    Z_22 = Z_22
    U = U
    beta0 = beta0
    beta_X1 = beta_X1
    beta_X2 = beta_X2
    beta_U = beta_U
    
    if np.random.random() > R1_prev:
        return 1
    else:
        return 0

def generate_z_star_fnc(Z, R_1) -> Tuple[np.ndarray]:
    
 Z_star_11 = Z[0] if R_1 == 1 else np.nan
           
 Z_star_12 = Z[1] if R_1 == 1 else np.nan
           
 Z_star_22 = Z[2] 

 return Z_star_11, Z_star_12, Z_star_22

def specify_betas_fnc() -> Tuple[np.ndarray]:
    
    beta0 = beta0
    beta_X1 = beta_X1
    beta_X2 = beta_X2
    beta_U = beta_U
    
    return beta0, beta_X1, beta_X2, beta_U

def generate_train_and_vali_data_fnc(N_train,
                                     N_vali,
                                     W,
                                     b,
                                     R1_prev,
                                     beta0,
                                     beta_X1,
                                     beta_X2,
                                     beta_U,
                                     gamma0,
                                     gamma_X1,
                                     gamma_X2,
                                     gamma_U,
                                     stddev) -> None:
       
    x_dim = 2
    z_dim = 3
    z_star_dim = 3
    u_dim = 1
    
    
    column_headers = [f'X_{i}' for i in range(x_dim)] + [f'Z_{i}' for i in range(z_dim)] + [f'Z_star_{i}' for i in range(z_star_dim)] + ['U', 'Y']
    print(f'Column names will be {column_headers}')
    #output_file = open(output_path, 'w')
   
    #output_file.write(','.join(column_headers) + '\n')
    train_data = []
    vali_data = []
    
    
    for n in range(N_train):
           X = generate_x_fnc()
           U = generate_u_fnc(u_dim)
           Z = generate_z_fnc(X_1 = X[0], X_2 = X[1], W = W, b=b, nonlinearity = 'relu', stddev = stddev)
           Y = generate_y_fnc(X_1 = X[0], X_2 = X[1], U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)
           R_1 = generate_R1_fnc(R1_prev, Z_11 = Z[0], Z_12 = Z[1], Z_22 = Z[2], U = U, beta0 = beta0, beta_X1 = beta_X1, beta_X2 = beta_X2, beta_U = beta_U)
           
           
           Z_star_11 = Z[0] if R_1 == 1 else np.nan
           
           Z_star_12 = Z[1] if R_1 == 1 else np.nan
           
           Z_star_22 = Z[2] 
           
           Z_star = np.array([Z_star_11, Z_star_12, Z_star_22])
    
        # Concatenate these together
           output_line_train = np.concatenate([X[0], X[1], U, Z, Z_star, Y]).tolist()
           train_data.append(output_line_train)
           #train_data = return_array.append(output_line)
        # Turn them into string-values
           #output_line = [str(val) for val in output_line]
        # Combine them into a comma-separated list
           #output_line = ','.join(output_line)
        # Write to file and add newline
           #output_file.write(output_line + '\n')
    
    for n in range(N_vali):
           X = generate_x_fnc()
           U = generate_u_fnc(u_dim)
           Z = generate_z_fnc(X_1 = X[0], X_2 = X[1], W = W, b=b, nonlinearity = 'relu', stddev = stddev)
           Y = generate_y_fnc(X_1 = X[0], X_2 = X[1], U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)
           R_1 = generate_R1_fnc(R1_prev, Z_11 = Z[0], Z_12 = Z[1], Z_22 = Z[2], U = U, beta0 = beta0, beta_X1 = beta_X1, beta_X2 = beta_X2, beta_U = beta_U)
           #Z_star = generate_z_star_fnc()
           
           Z_star_11 = Z[0] if R_1 == 1 else np.nan
           
           Z_star_12 = Z[1] if R_1 == 1 else np.nan
           
           Z_star_22 = Z[2] 
           
           Z_star = np.array([Z_star_11, Z_star_12, Z_star_22])
        
           output_line_vali = np.concatenate([X[0], X[1], U, Z, Z_star, Y]).tolist()
           vali_data.append(output_line_vali)
           #vali_data = return_array.append(output_line)
    #output_file.close()
    #print(f'Wrote {N} lines to {output_path}')
    return np.array(train_data), np.array(vali_data)

def mice_imputation_fnc(train_data, vali_data, **kwargs):
    
 mice_imputer = IterativeImputer(missing_values=np.nan, sample_posterior=True, **kwargs)
 mice_train_data = mice_imputer.fit_transform(train_data)
 mice_vali_data = mice_imputer.transform(vali_data)
    
 return mice_train_data, mice_vali_data

def mean_imputation_fnc(train_data, vali_data):
    
    mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose=0) #verbose = 0 means the columns and =1 means the rows

    mean_train_data = mean_imputer.fit_transform(train_data)
    mean_vali_data = mean_imputer.transform(vali_data)

    return mean_train_data, mean_vali_data

def calculate_target_measures_fnc(Y, predicted_risks):
    
    #-------------- Calculate Brier Score--------------------
    Brier = brier_score_loss(y_true = Y, y_prob = predicted_risks)
    
    #-------------- Calibration plot ------------------------
    Cal_Plot = calibration_curve(y_true = Y, y_prob = predicted_risks) #convert into slope and intercept rather than plot 
    
    
    #-------------- Calculate Observed VS Expected ratio-----
    observed_outcome = statistics.mean(Y)
    expected_outcome = statistics.mean(predicted_risks)
    O_E = observed_outcome / expected_outcome
    
    #-------------- AUROC -----------------------------------
    AUC = metrics.roc_auc_score(y_true = Y, y_score = predicted_risks)
    
    
    #-------------- create a dataframe with all the target measures--------
    target_measures = (Cal_Plot, O_E, AUC, Brier)
    
    #Target_Measures = pd.DataFrame(data = target_measures)
    
    
    return target_measures

def train_LR_and_get_predictions_fnc(mice_train_data, mice_vali_data, test_data):
    
    df_train = mice_train_data
    df_vali = mice_vali_data
    test_data = test_data
    
    #-------------- fit a LogReg to the imputed train data --------------------
    
    Z_star_train = df_train[[col for col in df_train if 'Z_star_' in col]].values
    Y_train = df_train['Y'].values
    print(f'Prevalence of Y_train: {Y_train.mean()}')
    
    clf = LogisticRegression(penalty='none', random_state=0)
    clf.fit(Z_star_train, Y_train)
    accuracy = clf.score(Z_star_train, Y_train)
    print(f'Accuracy using Z_train: {accuracy}')
    
    LR_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = LR_Y_hat_train)
    target_measures_train = {
        "dataset": "train",
        "model": "LR",
        "Cal_Plot": target_measures_train[0],
        "O_E": target_measures_train[1],
        "AUC": target_measures_train[2],
        "Brier": target_measures_train[3] 
    }
    #------------- evaluate the performance of LogReg on the mice imputed vali data -----------------
    
    Z_star_vali = df_vali[[col for col in df_vali if 'Z_star_' in col]].values
    Y_vali = df_vali['Y'].values
    print(f'Prevalence of Y_vali: {Y_vali.mean()}')
    
    #clf = LogisticRegression(penalty='none', random_state=0)
    #clf.fit(Z_star_vali, Y_vali)
    accuracy = clf.score(Z_star_vali, Y_vali)
    print(f'Accuracy using Z_vali: {accuracy}')
    
    LR_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = LR_Y_hat_vali)
    target_measures_vali = {
        "dataset": "vali",
        "model": "LR",
        "Cal_Plot": target_measures_vali[0],
        "O_E": target_measures_vali[1],
        "AUC": target_measures_vali[2],
        "Brier": target_measures_vali[3] 
    }
    #--------------- evaluate the performance of a LogReg on the mice imputed vali data ------------------
     
    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
    print(f'Prevalence of Y_test: {Y_test.mean()}')
    
    #clf = LogisticRegression(penalty='none', random_state=0)
    #clf.fit(Z_test, Y_test)
    accuracy = clf.score(Z_test, Y_test)
    print(f'Accuracy using Z_test: {accuracy}')
    
    LR_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = LR_Y_hat_test)
    target_measures_test = {
        "dataset": "test",
        "model": "LR",
        "Cal_Plot": target_measures_test[0],
        "O_E": target_measures_test[1],
        "AUC": target_measures_test[2],
        "Brier": target_measures_test[3] 
    }
    
    return [target_measures_train, target_measures_vali, target_measures_test]

def train_RF_and_get_predictions_fnc(train_data, vali_data, test_data):
    df_train = train_data
    df_vali = vali_data
    test_data = test_data
    
    #-------------- train a RandomForest on the train data --------------------
    Z_star_train = df_train[[col for col in df_train if 'Z_star_' in col]].values
    Y_train = df_train['Y'].values
    
    clf = RandomForestClassifier()
    clf.fit(Z_star_train, Y_train)
    
    RF_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = RF_Y_hat_train)
    target_measures_train = {
        "dataset": "train",
        "model": "RF",
        "Cal_Plot": target_measures_train[0],
        "O_E": target_measures_train[1],
        "AUC": target_measures_train[2],
        "Brier": target_measures_train[3] 
    }
    
    #-------------- evaluate the performance of a RandomForest on the validation data --------------------
    Z_star_vali = df_vali[[col for col in df_train if 'Z_star_' in col]].values
    Y_vali = df_vali['Y'].values
    
    #clf = RandomForestClassifier()
    #clf.fit(Z_star_vali, Y_vali)
    
    RF_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = RF_Y_hat_vali)
    target_measures_vali = {
        "dataset": "vali",
        "model": "RF",
        "Cal_Plot": target_measures_vali[0],
        "O_E": target_measures_vali[1],
        "AUC": target_measures_vali[2],
        "Brier": target_measures_vali[3] 
    }
    
    #-------------- evaluate the performance a RandomForest on the test data --------------------
    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
    
    #clf = RandomForestClassifier()
    #clf.fit(Z_test, Y_test)
    
    RF_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = RF_Y_hat_test)
    target_measures_test = {
        "dataset": "test",
        "model": "RF",
        "Cal_Plot": target_measures_test[0],
        "O_E": target_measures_test[1],
        "AUC": target_measures_test[2],
        "Brier": target_measures_test[3] 
    }
    
    
    return [target_measures_train, target_measures_vali, target_measures_test]

def train_BT_and_get_predictions_fnc(train_data, vali_data, test_data):
    df_train = train_data
    df_vali = vali_data
    test_data = test_data
    
    #-------------- train a RandomForest on the train data --------------------
    Z_star_train = df_train[[col for col in df_train if 'Z_star_' in col]].values
    Y_train = df_train['Y'].values
    
    clf = GradientBoostingClassifier()
    clf.fit(Z_star_train, Y_train)
    
    BT_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = BT_Y_hat_train)
    target_measures_train = {
        "dataset": "train",
        "model": "BT",
        "Cal_Plot": target_measures_train[0],
        "O_E": target_measures_train[1],
        "AUC": target_measures_train[2],
        "Brier": target_measures_train[3] 
    }
    
    #-------------- evaluate the performance a Gradient Boosting Tree on the validation data --------------------
    Z_star_vali = df_vali[[col for col in df_train if 'Z_star_' in col]].values
    Y_vali = df_vali['Y'].values
    
    BT_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = BT_Y_hat_vali)
    target_measures_vali = {
        "dataset": "vali",
        "model": "BT",
        "Cal_Plot": target_measures_vali[0],
        "O_E": target_measures_vali[1],
        "AUC": target_measures_vali[2],
        "Brier": target_measures_vali[3] 
    }
    #-------------- evaluate the performance of a Gradient Boosting Tree on the test data --------------------
    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
    
    BT_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = BT_Y_hat_test)
    target_measures_test = {
        "dataset": "test",
        "model": "BT",
        "Cal_Plot": target_measures_test[0],
        "O_E": target_measures_test[1],
        "AUC": target_measures_test[2],
        "Brier": target_measures_test[3] 
    }
    
    return [target_measures_train, target_measures_vali, target_measures_test]

def single_run_fnc(N_test,
                   N_train,
                   N_vali,
                   gamma0,
                   gamma_X1,
                   gamma_X2,
                   gamma_U, 
                   W,
                   b,
                   R1_prev,
                   beta0,
                   beta_X1,
                   beta_X2,
                   beta_U,
                   stddev):
    
    
    
    #W, b = turn_W_and_b_into_arrays(W11, W21, W21, ...)

    test_data = generate_test_data_fnc(N_test = N_test, W=W, b=b, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U, stddev = stddev)
    train_data, vali_data = generate_train_and_vali_data_fnc(N_train = N_train, N_vali = N_vali, W=W, b=b, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U, R1_prev = R1_prev, beta0 = beta0, beta_X1 = beta_X1, beta_X2 = beta_X2, beta_U = beta_U, stddev = stddev)
    
    
    mice_train_data, mice_vali_data = mice_imputation_fnc(train_data = train_data, vali_data = vali_data)
    mean_train_data, mean_vali_data = mean_imputation_fnc(train_data = train_data, vali_data = vali_data)
    
    x_train_data = train_data[..., :-1] #all the columns excluding Y
    x_vali_data = vali_data[..., :-1]
    x_mice_train_data, x_mice_vali_data = mice_imputation_fnc(x_train_data, x_vali_data, random_state=0, max_iter=10)

    # Concatenate imputed data and labels.
    mice_train_data = np.zeros_like(train_data)
    mice_train_data[..., :-1] = x_mice_train_data
    mice_train_data[..., -1] = train_data[..., -1]

    mice_vali_data = np.zeros_like(vali_data) #create an empty array to store the new dataset
    mice_vali_data[..., :-1] = x_mice_vali_data
    mice_vali_data[..., -1] = vali_data[..., -1]

    mean_train_data = pd.DataFrame(mean_train_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])
    mean_vali_data = pd.DataFrame(mean_vali_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])


    mice_train_data = pd.DataFrame(mice_train_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])
    mice_vali_data = pd.DataFrame(mice_vali_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])


    train_data = pd.DataFrame(mice_train_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])
    vali_data = pd.DataFrame(mice_vali_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','Z_star_1','Z_star_2','Z_star_3','U','Y'])
    test_data = pd.DataFrame(test_data, columns=['X_1','X_2','Z_1','Z_2','Z_3','U','Y'])    
    
    LR_target_measures = train_LR_and_get_predictions_fnc(mice_train_data = mice_train_data, mice_vali_data = mice_vali_data, test_data = test_data)
    RF_target_measures = train_RF_and_get_predictions_fnc(train_data = train_data, vali_data = vali_data, test_data = test_data)
    BT_target_measures = train_BT_and_get_predictions_fnc(train_data = train_data, vali_data = vali_data, test_data = test_data)
    
    Y = test_data['Y'].values
    
    return pd.DataFrame(LR_target_measures +  RF_target_measures + BT_target_measures)

single_run_fnc(
        N_test = 30000,
        N_train = 30000,
        N_vali = 10000,
        gamma0 = 0,
        gamma_X1 = 0.5,
        gamma_X2 = 0.5,
        gamma_U = 0.5, 
        W = np.array([[0.5, 0], [0.5, 0.5], [0, 0.5]]),
        b = np.array([1, 1, 1]),
        R1_prev = 0.5,
        beta0 = 0,
        beta_X1 = 0,
        beta_X2 = 0,
        beta_U = 0,
        stddev = 0.5
)

def save_simulation_parameters_and_target_measures_fnc(dag_type: str,
                                                       iteration: int,
                                                       parameters: dict,
                                                       target_measures: pd.DataFrame,
                                                       column_names: list[str],
                                                       file_path: str):
    """
    Add the DAG type, iteration, and all the parameters as new columns alongside the target measures.
    Then save to the file.
    """
    dataframe_to_save = target_measures.copy()
    dataframe_to_save["DAG type"] = dag_type
    dataframe_to_save["iteration"] = iteration
    for parameter_name, parameter_value in parameters.items():
        dataframe_to_save[parameter_name] = str(parameter_value)
    
    # Ensure that dataframe has all the expected columns in the expected order
    for column in column_names:
        if column not in dataframe_to_save.columns:
            print(f"Warning! We didn't find column {column}")
            dataframe_to_save[column] = "unknown"
    
    dataframe_to_save_ordered = dataframe_to_save.loc[:, column_names]
    assert (dataframe_to_save_ordered.columns == column_names).all()
    
    print("We're saving!")
    # Save to CSV
    dataframe_to_save_ordered.to_csv(file_path, mode="a", index=False, header=False)

def n_run_fnc(n_iterations = 1, output_file_path: str = "test_output.csv"):
                                                                                                                        
    # Define defaults
    default_parameters = {
        "N_test": 30000,
        "N_train": 30000,
        "N_vali": 10000,
        "gamma0": 0, 
        "gamma_X1": 1,
        "gamma_X2": 1,
        "gamma_U": 1,
        "W": np.array([[0.5, 0], [0.5, 0.5], [0, 0.5]]),
        "b": np.array([0, 0, 0]), 
        "R1_prev": 0.5,
        "stddev": 0.5
        }
    
     # Define column names for later saving, initialise file with header
    column_names = ["iteration",
                    "N_test", 
                    "N_train", 
                    "N_vali", 
                    "gamma0", 
                    "gamma_X1", 
                    "gamma_X2", 
                    "gamma_U", 
                    "W", 
                    "b", 
                    "R1_prev", 
                    "stddev",
                    "beta0",
                    "beta_X1",
                    "beta_X2",
                    "beta_U",
                    "dataset", 
                    "model", 
                    "Cal_Plot",
                    "O_E", 
                    "AUC", 
                    "Brier"]
    
    print(f"Saving outputs to {output_file_path}. Column names: {column_names}.")
    with open(output_file_path, "w") as f:
        f.write(",".join(column_names) + "\n")
    
    # Create MCAR parameters
    MCAR_parameter_setting = []
    # for beta_X1 in [0.5, 1]:
    #     for beta_X2 in [0.5, 1]:
    #         for beta_U in [0.5, 1]:
    single_MCAR_parameter_setting = {"beta0": 0, "beta_X1": 0, "beta_X2": 0, "beta_U": 0}
    MCAR_parameter_setting.append(single_MCAR_parameter_setting)
    
    
    # Create MAR parameters beta_x2 will be non-zero
    MAR_parameter_setting = []
    for beta_X2 in [0.5, 1]:
        single_MAR_parameter_setting = {"beta0": 0, "beta_X1": 0, "beta_X2": beta_X2, "beta_U": 0}
        MAR_parameter_setting.append(single_MAR_parameter_setting)

   
   # Create MNAR1 parameters
   # Beta_X1 should be non-zero
    MNAR1_parameter_setting = []
    for beta_X1 in [0.5, 1]:
        single_MNAR1_parameter_setting = {"beta0": 0, "beta_X1": beta_X1, "beta_X2": 0, "beta_U": 0}
        MNAR1_parameter_setting.append(single_MNAR1_parameter_setting)
    # MNAR1_parameter_setting should have 2 entries in it !!
    assert len(MNAR1_parameter_setting) == 2
       

   # Create MNAR2 parameters
   # beta_U is non-zero
    MNAR2_parameter_setting = []
    for beta_U in [0.5, 1]: 
        single_MNAR2_parameter_setting = {"beta0": 0, "beta_X1": 0, "beta_X2": 0, "beta_U": beta_U}
        MNAR2_parameter_setting.append(single_MNAR2_parameter_setting)
    # This list should have 2 values
    assert len(MNAR2_parameter_setting) == 2


   # Create MNAR3 parameters
   # Beta_X1, Beta_X2, and Beta_U are non-zero
    MNAR3_parameter_setting = []
    for beta_X1 in [0.5, 1]:
        for beta_X2 in [0.5, 1]:
            for beta_U in [0.5, 1]: 
                single_MNAR3_parameter_setting = {"beta0": 0, "beta_X1": beta_X1, "beta_X2": beta_X2, "beta_U": beta_U}
                MNAR3_parameter_setting.append(single_MNAR3_parameter_setting)
    # This list should have 8 values in it
    assert len(MNAR3_parameter_setting) == 8


    # Create a list with all the Ws of interest
    W_options = [] # possibly use product here

    W_20 = 0
    W_01 = 0
    W_entry_options = [0.5, 1.0]
    for W_00 in W_entry_options:
        for W_10 in W_entry_options:
            for W_11 in W_entry_options:
                for W_21 in W_entry_options:
                    W = np.array([[W_00, W_01], [W_10, W_11], [W_20, W_21]])
                    W_options.append(W)
    # this should have 16 entries in it
    assert len(W_options) == 16

#create dictionaries to vary individual parameters excluding betas which are part of the DAG scenarios above (eg W, R1_prev, stddev, gammas)

    non_beta_parameter_options = []
    for gamma_X1 in [0.5, 1]:
        for gamma_X2 in [0.5, 1]:
            for gamma_U in [0.5, 1]:
                for W in W_options:
                    for b in [np.array([0, 0, 0]), np.array([1, 1, 1])]:
                                parameters = {
                                              "gamma_X1": gamma_X1, 
                                              "gamma_X2": gamma_X2, 
                                              "gamma_U": gamma_U, 
                                              "W": W, 
                                              "b": b, 
                                              }
                                non_beta_parameter_options.append(parameters)
                                # assert len(non_beta_parameter_options) == 144 * 16 #2304
    # Now JUST vary stddev
    for stddev in [0.5, 1]:
        parameters = {"stddev": stddev}
        non_beta_parameter_options.append(parameters)

    for R1_prev in [0.1, 0.2, 0.5]:
        parameteres = {"R1_prev": R1_prev}
        non_beta_parameter_options.append(parameteres)
    


    # Combine all the scenarios
    DAG_settings_to_explore = {"MCAR": MCAR_parameter_setting,
                               "MAR": MAR_parameter_setting,
                               "MNAR1": MNAR1_parameter_setting,
                               "MNAR2": MNAR2_parameter_setting,
                               "MNAR3": MNAR3_parameter_setting}
    [
         MCAR_parameter_setting, 
         MAR_parameter_setting, 
         MNAR1_parameter_setting, 
         MNAR2_parameter_setting, 
         MNAR3_parameter_setting
         ]

    print(DAG_settings_to_explore)

    # Two things to be aware of:
    # - DAG_settings_to_explore is current a LIST of LISTS
    # - We are passing a dictionary to single_run_fnc but it expects a bunch of parameters
    # - Target measures is a Pandas dataframe
    # How many simulations are we going to run?

    total_dag_settings = sum([len(dag_setting_list) for dag_setting_list in DAG_settings_to_explore.values()])
    total_simulations = total_dag_settings * len(non_beta_parameter_options) * n_iterations
    print(f"About to run {total_simulations} simulations")

    # return False
    
    # For each DAG setting in our list of DAG settings:
    for DAG_setting_name, DAG_setting_list in DAG_settings_to_explore.items():
        print(f"Exploring DAGs of the type {DAG_setting_name}")
        for DAG_setting in DAG_setting_list:
            # Loop over all the "non-beta parameters" to vary
            for non_beta_parameter_setting in non_beta_parameter_options:       # we have 288 here, so 17 * 288 in total
                # Create the full set of parameters to give to the simulation
                # We start with the default parameters (dictionary)
                this_simulation_parameters = {k: v for k, v in default_parameters.items()}
                # We use the DAG_setting to set the betas
                for parameter, new_value in DAG_setting.items():
                    print(f"Setting parameter {parameter} to be value {new_value}")
                    this_simulation_parameters[parameter] = new_value
                # We use the non_beta_paramter_setting to set everything else
                for parameter, new_value in non_beta_parameter_setting.items():
                    print(f"Setting parameter {parameter} to be value {new_value}")
                    this_simulation_parameters[parameter] = new_value
                print(f"Running simulation with parameters {this_simulation_parameters}")

                # And run the simulation for those
                for iteration in range(n_iterations):
                    t0 = time()
                    target_measures = single_run_fnc(
                                                    N_test = this_simulation_parameters["N_test"],
                                                    N_train = this_simulation_parameters["N_train"],
                                                    N_vali = this_simulation_parameters["N_vali"],
                                                    gamma0 = this_simulation_parameters["gamma0"],
                                                    gamma_X1 = this_simulation_parameters["gamma_X1"],
                                                    gamma_X2 = this_simulation_parameters["gamma_X2"],
                                                    gamma_U = this_simulation_parameters["gamma_U"],
                                                    beta0 = this_simulation_parameters["beta0"],
                                                    beta_X1 = this_simulation_parameters["beta_X1"],
                                                    beta_X2 = this_simulation_parameters["beta_X2"],
                                                    beta_U = this_simulation_parameters["beta_U"],
                                                    W = this_simulation_parameters["W"],
                                                    b = this_simulation_parameters["b"],
                                                    R1_prev = this_simulation_parameters["R1_prev"],
                                                    stddev = this_simulation_parameters["stddev"]
                                                    )
                    elapsed_time = time() - t0
                    print(f"Single iteration took {elapsed_time} seconds")
                
                    print(this_simulation_parameters)
                
                    
                    # Save simulation parameters and target measures to file
                    save_simulation_parameters_and_target_measures_fnc(dag_type = DAG_setting_name,
                                                                       column_names = column_names,             
                                                                       iteration = iteration,
                                                                       parameters = this_simulation_parameters,
                                                                       target_measures = target_measures,
                                                                       file_path = output_file_path)
                   

        

n_run_fnc()


cwd = os.getcwd()

print(cwd)

