from re import X
from sre_constants import NOT_LITERAL_IGNORE
from tkinter import N
import pandas as pd
import numpy as np
from typing import List
from typing import Dict
from pathlib import Path
import torch
import statsmodels.api as sm
from torch import relu, tanh, sigmoid
from typing import Tuple
from typing import Dict
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import brier_score_loss
from sklearn import metrics
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from time import time
from tqdm import tqdm
import argparse 
from scipy.special import expit
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Logit

def generate_x_fnc(N) -> np.ndarray:
    X_1 = np.random.normal(loc = 1, size = N)
    X_2 = np.random.normal(loc = 1, size = N) #loc (mean) should be higher because we want X to be distributed around positive numbers
    
    return X_1, X_2

def generate_u_fnc(N) -> np.ndarray:
    U = np.random.normal(size = N)
    return U

def generate_y_fnc(X_1, X_2, U, gamma_X1, gamma_X2, gamma_U, gamma0) -> int:
    p_y = sigmoid(torch.tensor(gamma0 + gamma_X1 * X_1 + gamma_X2 * X_2 + gamma_U * U)).numpy()
    assert (0 < p_y).all() and (p_y < 1).all() #I added .all() as I was getting an error, but not sure if that's ok now? 
    Y = np.random.binomial(n = 1, p = p_y)
    return Y

def generate_z_fnc(X_1, X_2, W, b, stddev, nonlinearity: str = 'relu') -> np.ndarray:
    
    X = np.stack([X_1, X_2], axis = 1)
    
    print(X.shape)
    print(W.shape)
    
    in_dim = X.shape[1]
    out_dim = W.shape[0]
    
    assert W.shape == (out_dim, in_dim)
    assert b.shape[0] == out_dim
    
    pre_nonlinearity = np.matmul(X, W.T) + b

    if nonlinearity == 'relu':
        Z_mean = relu(torch.tensor(pre_nonlinearity)).numpy()
    elif nonlinearity == 'tanh':
        Z_mean = tanh(torch.tensor(pre_nonlinearity)).numpy()
    else:
        raise NotImplementedError(f'Requested nonlinearity {nonlinearity} is not implemented yet.')
    print(Z_mean.shape)
    Z = np.random.normal(loc=Z_mean, scale=stddev)
 
    assert Z.shape[1] == out_dim
    
    return Z


def generate_R1_fnc(R1_prev, beta0, beta_X1, beta_X2, beta_U, X_1, X_2, U) -> np.ndarray:
    R1_prev = R1_prev
    beta0 = beta0
    beta_X1 = beta_X1
    beta_X2 = beta_X2
    beta_U = beta_U
    X_1 = X_1
    X_2 = X_2
    U = U
    
    
    # TODO: R should depend on the betas, not just the R prevalence!- update: check if the code below is correct? 
    
    binomial = sm.families.Binomial(sm.families.links.logit())
    print(X_1.shape, X_2.shape, U.shape,"test")
    pla = np.random.binomial(1, R1_prev, size=300000)
    print(pla.shape)
    beta0 = sm.GLM(np.random.binomial(1, R1_prev, size=300000),
            sm.add_constant((beta_X1 * X_1 + beta_X2 * X_2 + beta_U * U)),
            family=binomial).fit().params[0]

    R_1 = np.random.binomial(1, p=expit(beta0 + beta_X1 * X_1 + beta_X2 * X_2 + beta_U * U), size=300000)
 

    return R_1


def generate_z_star_fnc(Z, R_1) -> Tuple[np.ndarray]:
    Z_star_11 = Z[0] if R_1 == 1 else np.nan
    Z_star_12 = Z[1] if R_1 == 1 else np.nan
    Z_star_22 = Z[2] 

    return Z_star_11, Z_star_12, Z_star_22

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

    column_headers = [f'X_{i}' for i in range(x_dim)] + [f'Z_{i}' for i in range(z_dim)] + ['U', 'Y']
    print(f'Test data column names will be {column_headers}')
    print(f"Generating {N_test} test data points...")

    test_data = {c: [] for c in column_headers}

    X_1, X_2 = generate_x_fnc(N = N_test)
    U = generate_u_fnc(N = N_test)
    Z = generate_z_fnc(X_1 = X_1, X_2 = X_2, W = W, b = b, nonlinearity = 'relu', stddev = stddev)
    Y = generate_y_fnc(X_1 = X_1, X_2 = X_2, U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)

    test_data = pd.DataFrame({"X_0": X_1, "X_1": X_2, "U": U, "Z_1": Z[:,0], "Z_2": Z[:,1], "Z_3": Z[:,2], "Y": Y})
    print(test_data.head())
    return test_data


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



    column_headers = [f'X_{i}' for i in range(x_dim)] + [f'Z_{i}' for i in range(z_dim)] + [f'Z_star_{i}' for i in range(z_star_dim)] + ['R_1', 'U', 'Y']
    print(f"Generating {N_train} training and {N_vali} validation data points")
    print(f'Train and vali data column names will be {column_headers}')

    train_data = {c: [] for c in column_headers}
    vali_data = {c: [] for c in column_headers}

    
    X_1, X_2 = generate_x_fnc(N = N_train)
    U = generate_u_fnc(N = N_train)
    Z = generate_z_fnc(X_1 = X_1, X_2 = X_2, W = W, b=b, nonlinearity = 'relu', stddev = stddev)
    Y = generate_y_fnc(X_1 = X_1, X_2 = X_2, U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)
    R_1 = generate_R1_fnc(R1_prev = R1_prev, beta0 = beta0, beta_X1 = beta_X1, beta_X2 = beta_X2, beta_U = beta_U, X_1 = X_1, X_2 = X_2, U = U)

   

    Z_star_11 = Z[:,0] if R_1.all() == 1 else np.nan 
    Z_star_12 = Z[:,1] if R_1.all() == 1 else np.nan
    Z_star_22 = Z[:,2] 
    Z_star = np.array([Z_star_11, Z_star_12, Z_star_22])

    X = np.stack([X_1, X_2], axis = 1)

    for i in range(x_dim): #we still want to keep this 
            train_data[f"X_{i}"].append(X[i])
    for i in range(z_dim):
            train_data[f"Z_{i}"].append(Z[i])
    for i in range(z_star_dim):
            train_data[f"Z_star_{i}"].append(Z_star[i])
    train_data["U"].append(U)
    train_data["Y"].append(Y)
    train_data["R_1"].append(R_1)



    X = generate_x_fnc(N = N_train)
    U = generate_u_fnc(N = N_train)
    Z = generate_z_fnc(X_1 = X_1, X_2 = X_2, W = W, b=b, nonlinearity = 'relu', stddev = stddev)
    Y = generate_y_fnc(X_1 = X_1, X_2 = X_2, U = U, gamma0 = gamma0, gamma_X1 = gamma_X1, gamma_X2 = gamma_X2, gamma_U = gamma_U)
    R_1 = generate_R1_fnc(R1_prev = R1_prev, beta0 = beta0, beta_X1 = beta_X1, beta_X2 = beta_X2, beta_U = beta_U, X_1=X_1, X_2=X_2, U = U)

    Z_star_11 = Z[0] if R_1.all() == 1 else np.nan
    Z_star_12 = Z[1] if R_1.all() == 1 else np.nan
    Z_star_22 = Z[2] 
    Z_star = np.array([Z_star_11, Z_star_12, Z_star_22])

    for i in range(x_dim):
            vali_data[f"X_{i}"].append(X[i])
    for i in range(z_dim):
            vali_data[f"Z_{i}"].append(Z[i])
    for i in range(z_star_dim):
            vali_data[f"Z_star_{i}"].append(Z_star[i])
    vali_data["U"].append(U)
    vali_data["Y"].append(Y)
    vali_data["R_1"].append(R_1)
    
    train_data = pd.DataFrame({"X_0": X_1, "X_1": X_2, "U": U, "Z_1": Z[:,0], "Z_2": Z[:,1], "Z_3": Z[:,2], "R_1": R_1, "Y": Y})
    vali_data = pd.DataFrame({"X_0": X_1, "X_1": X_2, "U": U, "Z_1": Z[:,0], "Z_2": Z[:,1], "Z_3": Z[:,2], "R_1": R_1, "Y": Y})
    
    return train_data, vali_data


def mean_imputation_fnc(train_data, vali_data):

    mean_imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean') #verbose = 0 means the columns and =1 means the rows (warning msg, remove verbose)

    mean_train_data = mean_imputer.fit_transform(train_data)
    mean_vali_data = mean_imputer.transform(vali_data)
    print("Missing values in training data after imputation:\n", np.isnan(mean_train_data).sum())
    print("Missing values in validation data after imputation:\n", np.isnan(mean_vali_data).sum())

    return mean_train_data, mean_vali_data

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def calculate_target_measures_fnc(Y, predicted_risks):
    


    #-------------- Calculate Brier Score--------------------
    Brier = brier_score_loss(y_true = Y, y_prob = predicted_risks)

    #-------------- AUC-----------------------------------
    AUC = metrics.roc_auc_score(y_true = Y, y_score = predicted_risks)
    
    #-------------- calibration intercept ----
    
    predicted_risks = np.clip(predicted_risks, 0.1, 0.9)
    LP = np.log(predicted_risks / (1 - predicted_risks))
    binomial = sm.families.Binomial(sm.families.links.logit())
    Cal_Int = sm.GLM(Y, sm.add_constant(LP), family=binomial).fit().params[0]
    #-------------- calibration slope ----
    Cal_Slope = sm.GLM(Y, sm.add_constant(LP), family=binomial).fit().params[1]
    #-------------- create a dataframe with all the target measures--------
    target_measures = {
        "AUC": AUC,
        "Brier": Brier,
        "Cal_Int": Cal_Int,
        "Cal_Slope": Cal_Slope
    }
    
    return target_measures


def train_LR_and_get_predictions_fnc(mean_train_data, mean_vali_data, test_data) -> pd.DataFrame:
    print(f"Training Logistic Regression model...")

    df_train = mean_train_data
    df_vali = mean_vali_data
    test_data = test_data
    
    
    #-------------- train LogReg with the imputed train data --------------------
    print(df_train.head())
    Z_star_train = df_train[[col for col in df_train if 'Z_' in col]].values
    print(Z_star_train, "where is Z_star") 
    Y_train = df_train['Y'].values
    print(f'Prevalence of Y_train: {Y_train.mean()}')

    clf = LogisticRegression(penalty='none', random_state=0)
    clf.fit(Z_star_train, Y_train)
    accuracy = clf.score(Z_star_train, Y_train)
    print(f'Accuracy using Z_train: {accuracy}')

    LR_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y=Y_train, predicted_risks=LR_Y_hat_train)
    target_measures_train["split"] = "train"
    
    #------------- evaluate the performance of LogReg on the mean imputed vali data -----------------
    
    Z_star_vali = df_vali[[col for col in df_vali if 'Z_' in col]].values
    Y_vali = df_vali['Y'].values
    print(f'Prevalence of Y_vali: {Y_vali.mean()}')

    accuracy = clf.score(Z_star_vali, Y_vali)
    print(f'Accuracy using Z_vali: {accuracy}')

    LR_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y=Y_vali, predicted_risks=LR_Y_hat_vali)
    target_measures_vali["split"] = "vali"# TODO add this to the other models

    #--------------- evaluate the performance of a LogReg on test data ------------------

    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
    print(f'Prevalence of Y_test: {Y_test.mean()}')

    accuracy = clf.score(Z_test, Y_test)
    print(f'Accuracy using Z_test: {accuracy}')

    LR_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y=Y_test, predicted_risks=LR_Y_hat_test)
    target_measures_test["split"] = "test"      # TODO add this to the other models

    # Combine all the LR results - TODO make the other models follow this format
    lr_results = pd.DataFrame([target_measures_train, target_measures_vali, target_measures_test])
    lr_results["model"] = "Logistic Regression"# TODO add this to the other models

    return lr_results

#-----------------------------------------------------------------------------------------------------------------

def train_RF_and_get_predictions_fnc(mean_train_data, mean_vali_data, test_data):
    print(f"Training Random Forest model...")
    df_train = mean_train_data
    df_vali = mean_vali_data
    test_data = test_data
    
    #-------------- train a RandomForest on the train data --------------------
    Z_star_train = df_train[[col for col in df_train if 'Z_' in col]].values
    Y_train = df_train['Y'].values
    
    clf = RandomForestClassifier(max_depth=5)
    clf.fit(Z_star_train, Y_train)
    
    RF_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = RF_Y_hat_train)
    target_measures_train["split"] = "train"
    
    #-------------- evaluate the performance of a RandomForest on the validation data --------------------
    Z_star_vali = df_vali[[col for col in df_vali if 'Z_' in col]].values
    Y_vali = df_vali['Y'].values
    
    RF_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = RF_Y_hat_vali)
    target_measures_vali["split"] = "vali"
    
    #-------------- evaluate the performance a RandomForest on the test data --------------------
    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
 
    RF_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = RF_Y_hat_test)
    target_measures_test["split"] = "test"

    # Combine all the RF results - TODO make the other models follow this format
    rf_results = pd.DataFrame([target_measures_train, target_measures_vali, target_measures_test])
    rf_results["model"] = "Random Forest"# TODO add this to the other models
    return rf_results


def train_BT_and_get_predictions_fnc(train_data, vali_data, test_data):
    print(f"Training  Gradient Boosting Tree model...")
    df_train = train_data
    df_vali = vali_data
    test_data = test_data
    
    #-------------- train a RandomForest on the train data --------------------
    Z_star_train = df_train[[col for col in df_train if 'Z_' in col]].values
    Y_train = df_train['Y'].values
    
    clf = HistGradientBoostingClassifier()
    clf.fit(Z_star_train, Y_train)
    
    BT_Y_hat_train = clf.predict_proba(Z_star_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = BT_Y_hat_train)
    target_measures_train["split"] = "train"
    
    #-------------- evaluate the performance a Gradient Boosting Tree on the validation data --------------------
    Z_star_vali = df_vali[[col for col in df_vali if 'Z_' in col]].values
    Y_vali = df_vali['Y'].values
    
    BT_Y_hat_vali = clf.predict_proba(Z_star_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = BT_Y_hat_vali)
    target_measures_vali["split"] = "vali"
   
    #-------------- evaluate the performance of a Gradient Boosting Tree on the test data --------------------
    Z_test = test_data[[col for col in test_data if 'Z_' in col]].values
    Y_test = test_data['Y'].values
    
    BT_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = BT_Y_hat_test)
    target_measures_test["split"] = "test"

    # Combine all the BT results - TODO make the other models follow this format
    bt_results = pd.DataFrame([target_measures_train, target_measures_vali, target_measures_test])
    bt_results["model"] = "Gradient Boosting Tree"# TODO add this to the other models
    return bt_results 
   #-----------------------------------------------------------------------------------------------------------------------------------------------------------

def train_MLP_and_get_predictions_fnc(train_data, vali_data, test_data):
    print(f"Training MLP model...")
    df_train = train_data
    df_vali = vali_data
    test_data = test_data
    
    #-------------- train a MLP neural network on the train data --------------------
    Z_star_and_R_cols = [col for col in df_train if 'Z_' in col] + ['R_1']
    Z_train = df_train[Z_star_and_R_cols].values
    Y_train = df_train['Y'].values

    clf = MLPClassifier(hidden_layer_sizes=(4, 7), max_iter=500, alpha=0.0001, solver='adam', 
                        random_state=42, tol=0.0001)
    clf.fit(Z_train, Y_train)
    
    MLP_Y_hat_train = clf.predict_proba(Z_train)[:, 1]
    target_measures_train = calculate_target_measures_fnc(Y = Y_train, predicted_risks = MLP_Y_hat_train)
    target_measures_train["split"] = "train"

    #-------------- evaluate the performance of a MLP neural network on the validation data --------------------
    Z_vali = df_vali[Z_star_and_R_cols].values
    Y_vali = df_vali['Y'].values

    MLP_Y_hat_vali = clf.predict_proba(Z_vali)[:, 1]
    target_measures_vali = calculate_target_measures_fnc(Y = Y_vali, predicted_risks = MLP_Y_hat_vali)
    target_measures_vali["split"] = "vali"
    
    #-------------- evaluate the performance of a MLP neural network on the test data --------------------
    # Add a column of 1s to the test data
    test_data["R_1"] = 1     # TODO check: does R_1 = 1 mean observed or missing? R_1 = 0 = X1 missing | R_1 = 1 = X1 present
    Z_test = test_data[["Z_1", "Z_2", "Z_3", "R_1"]].values
    Y_test = test_data['Y'].values

    MLP_Y_hat_test = clf.predict_proba(Z_test)[:, 1]
    target_measures_test = calculate_target_measures_fnc(Y = Y_test, predicted_risks = MLP_Y_hat_test)
    target_measures_test["split"] = "test"

    mlp_results = pd.DataFrame([target_measures_train, target_measures_vali, target_measures_test])
    mlp_results["model"] = "MLP"
    return mlp_results


#-------------------------------------------------------------------------------------------------------------------------------------------------------------

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

    test_data = generate_test_data_fnc(
        N_test = N_test, 
        W=W, b=b, 
        gamma0 = gamma0, 
        gamma_X1 = gamma_X1, 
        gamma_X2 = gamma_X2, 
        gamma_U = gamma_U, 
        stddev = stddev)
    
    train_data, vali_data = generate_train_and_vali_data_fnc(
        N_train = N_train, 
        N_vali = N_vali, 
        W=W, 
        b=b, 
        R1_prev = R1_prev, 
        gamma0 = gamma0, 
        gamma_X1 = gamma_X1, 
        gamma_X2 = gamma_X2, 
        gamma_U = gamma_U, 
        beta0 = beta0, 
        beta_X1 = beta_X1, 
        beta_X2 = beta_X2, 
        beta_U = beta_U, 
        stddev = stddev)

    mean_train_data, mean_vali_data = mean_imputation_fnc(
        train_data = train_data, 
        vali_data = vali_data)

    mean_train_data = pd.DataFrame(mean_train_data, columns=train_data.columns)
    mean_vali_data = pd.DataFrame(mean_vali_data, columns=vali_data.columns)

    LR_target_measures = train_LR_and_get_predictions_fnc(mean_train_data = mean_train_data, mean_vali_data = mean_vali_data, test_data = test_data)
    RF_target_measures = train_RF_and_get_predictions_fnc(mean_train_data = mean_train_data, mean_vali_data = mean_vali_data, test_data = test_data)
    BT_target_measures = train_BT_and_get_predictions_fnc(train_data = train_data, vali_data = vali_data, test_data = test_data)
    MLP_target_measures = train_MLP_and_get_predictions_fnc(train_data = mean_train_data, vali_data = mean_vali_data, test_data = test_data)

    all_results = pd.concat([LR_target_measures, RF_target_measures, BT_target_measures, MLP_target_measures], axis=0)
    return all_results


def save_simulation_parameters_and_target_measures_fnc(dag_type: str,
                                                       iteration: int,
                                                       parameters: dict,
                                                       target_measures: pd.DataFrame,
                                                       column_names: List[str],
                                                       file_path: str):
    """
    Add the DAG type, iteration, and all the parameters as new columns alongside the target measures.
    Then save to the file.
    """
    dataframe_to_save = target_measures.copy()
    dataframe_to_save["dag_type"] = dag_type
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


def get_all_parameter_settings() -> Tuple[List[Dict], List[str]]:
    """
    Return a list of all the parameter settings we care about.
    """
    # Define defaults
    default_parameters = {
        "N_test": 300000,
        "N_train": 300000,
        "N_vali": 100000,
        "gamma0": 0, 
        "gamma_X1": 0.5,
        "gamma_X2": 0.5,
        "gamma_U": 0.5,
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
                    "split", 
                    "dag_type",
                    "model", 
                    "Cal_Int",
                    "Cal_Slope",
                    "AUC", 
                    "Brier"]

    # Create MCAR parameters
    MCAR_parameter_setting = []
    for beta_X1 in [0.5, 1]:
        for beta_X2 in [0.5, 1]:
            for beta_U in [0.5, 1]:
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
    # for gamma_X1 in [0.5, 1]:
    #     for gamma_X2 in [0.5, 1]:
    #         for gamma_U in [0.5, 1]:
    for W in W_options:
        for b in [np.array([0, 0, 0]), np.array([1, 1, 1])]:
            parameters = {
            "gamma_X1": 0.5, 
            "gamma_X2": 0.5, 
            "gamma_U": 0.5, 
            "W": W, 
            "b": b, 
            }
            non_beta_parameter_options.append(parameters)
                                # assert len(non_beta_parameter_options) == 144 * 16 #2304
    # Now JUST vary stddev
    for stddev in [0.5, 1]:
        parameters = {"stddev": stddev}
        non_beta_parameter_options.append(parameters)

    for R1_prev in [0.2, 0.5, 0.8]:
        parameters = {"R1_prev": R1_prev}
        non_beta_parameter_options.append(parameters)
    
    # Combine all the scenarios
    DAG_settings_to_explore = {"MCAR": MCAR_parameter_setting,
                               "MAR": MAR_parameter_setting,
                               "MNAR1": MNAR1_parameter_setting,
                               "MNAR2": MNAR2_parameter_setting,
                               "MNAR3": MNAR3_parameter_setting}
    #print(DAG_settings_to_explore)

    all_parameter_settings = []
    for DAG_setting_name, DAG_setting_list in tqdm(DAG_settings_to_explore.items()):
        for DAG_setting in DAG_setting_list:
            # Loop over all the "non-beta parameters" to vary
            for non_beta_parameter_setting in non_beta_parameter_options:       # we have 288 here, so 17 * 288 in total
                # Create the full set of parameters to give to the simulation
                # We start with the default parameters (dictionary)
                this_simulation_parameters = {k: v for k, v in default_parameters.items()}
                # We use the DAG_setting to set the betas
                for parameter, new_value in DAG_setting.items():
                    this_simulation_parameters[parameter] = new_value
                # We use the non_beta_paramter_setting to set everything else
                for parameter, new_value in non_beta_parameter_setting.items():
                    this_simulation_parameters[parameter] = new_value
                # Add this simulation to the list of simulations to run
                this_simulation_parameters["DAG_setting_name"] = DAG_setting_name
                all_parameter_settings.append(this_simulation_parameters)

    print(f"Constructed a list of {len(all_parameter_settings)} simulations to run")
    return all_parameter_settings, column_names


def run_single_simulation(parameter_setting: Dict, n_iterations: int, column_names: List[str], output_file_path: Path) ->  Dict: 
    print(n_iterations)
    for iteration in tqdm(range(n_iterations)):
        t0 = time()
        target_measures = single_run_fnc(
                                        N_test = parameter_setting["N_test"],
                                        N_train = parameter_setting["N_train"],
                                        N_vali = parameter_setting["N_vali"],
                                        gamma0 = parameter_setting["gamma0"],
                                        gamma_X1 = parameter_setting["gamma_X1"],
                                        gamma_X2 = parameter_setting["gamma_X2"],
                                        gamma_U = parameter_setting["gamma_U"],
                                        beta0 = parameter_setting["beta0"],
                                        beta_X1 = parameter_setting["beta_X1"],
                                        beta_X2 = parameter_setting["beta_X2"],
                                        beta_U = parameter_setting["beta_U"],
                                        W = parameter_setting["W"],
                                        b = parameter_setting["b"],
                                        R1_prev = parameter_setting["R1_prev"],
                                        stddev = parameter_setting["stddev"]
                                        )
        elapsed_time = time() - t0
        print(f"Single iteration took {elapsed_time} seconds")
        DAG_setting_name = parameter_setting["DAG_setting_name"]

        # Save simulation parameters and target measures to file
        save_simulation_parameters_and_target_measures_fnc(dag_type = DAG_setting_name,
                                                           column_names = column_names,             
                                                           iteration = iteration,
                                                           parameters = parameter_setting,
                                                           target_measures = target_measures,
                                                           file_path = output_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter_setting", type=int, default=0)
    parser.add_argument("--n_iterations", type=int, default=2)
    args = parser.parse_args()
    print(f"Running parameter setting {args.parameter_setting} for {args.n_iterations} iterations.")

    all_parameter_settings, column_names = get_all_parameter_settings()
    total_simulations = len(all_parameter_settings)
    if args.parameter_setting >= total_simulations:
        raise ValueError(f"Invalid parameter setting: parameter_setting must be between 0 and {total_simulations - 1}")

    simulation_output_file_path = f"output_setting_{args.parameter_setting}.csv"
    this_parameter_setting = all_parameter_settings[args.parameter_setting]

    # Save the header
    with open(simulation_output_file_path, "w") as f:
        f.write(",".join(column_names) + "\n")

    run_single_simulation(this_parameter_setting,
                          n_iterations=args.n_iterations,
                          column_names=column_names,
                          output_file_path=simulation_output_file_path)