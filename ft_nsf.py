# %%
import sys
import os
import chardet
import warnings
import argparse
warnings.filterwarnings('ignore')

# %%
import pandas as pd
import numpy as np
import scipy as sp
import xgboost as xgb
import multiprocessing
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import preprocessing

from pandas.api.types import is_numeric_dtype
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import VarianceThreshold, chi2, f_regression, r_regression, RFECV, RFE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import linear_model, tree, ensemble
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import plot_tree


### **Functions**
#################

today = datetime.today()

def melt_ntop(df, ntop=5, column = "first_gift_fund"):

    # make everything lower case
    if not is_numeric_dtype(df[column]):
        df[column] = df[column].str.lower()
    
    # Filter out rows with NA values in column
    threshold = len(df) * 0.05

    df_ntop = df.dropna(subset=[column]) \
        .groupby(column) \
        .filter(lambda x: len(x) > threshold) \
        .groupby(column) \
        .size() \
        .sort_values(ascending=False) \
        .reset_index(name='count')
    
    # Get the top 5 funds
    top_vars = df_ntop[column].tolist()    
    
    # Create new columns for each of the top 5 funds
    for var in top_vars:
        var_str = str(var).replace(".", "_").replace('@', '_').replace(' ', '_')
        var_column_name = "%s_%s_binary" %(column, var_str)
        var_column_name = var_column_name.replace('__', '_')
        df[var_column_name] = df[column].apply(lambda x: 1 if x == var else 0)

    df = df.drop(columns=column)

    return(df)

def calculate_age(birth_date):
    if isinstance(birth_date, str):
        birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
        today = datetime.now()
        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        fraction = (today - birth_date.replace(year=today.year)).days / 365.25
        age_decimal = age + fraction
        return age_decimal
    else:
        return np.nan


def bin_and_convert_to_binary(ages):

    # Define bin edges
    bin_edges = [0, 30, 40, 50]

    # Bin the ages into groups
    binned_ages = np.digitize(ages, bins=bin_edges)
    
    # Convert binned ages into binary representation
    binary_ages = np.eye(len(bin_edges) + 1)[binned_ages]
    
    return binary_ages


### Main
########

DEFAULT_COLUMNS_TO_DROP = ["Unnamed: 0", "org_name", "wealth_screen_data", "attributes"]

def main(verbose=0):

    ### **Constituent File**
    ########################

    if verbose==1:
        print("Reading the constituent file")

    client = "national_scleroderma_foundation"
    path = "/home/RMittal@ccsfundraising.com/ccs_pred_mod"
    filename =  "%s_constituent_ccsdb.csv" %(client)
    myfile = "%s/%s" %(path, filename)
    df_cd = pd.read_csv(myfile, encoding="ISO-8859-1")


    ### **Drop unwanted columns**
    #############################

    if verbose==1:
        print("Dropping unwanted columns")

    #columns_to_drop=["Unnamed: 0", "org_name", "wealth_screen_data", "attributes"]
    if columns_to_drop:
        for col in columns_to_drop:
            if col in df_cd.keys().to_list():
                df_cd = df_cd.drop(columns=col)


    ### **Spouse**
    ##############

    if verbose==1:
        print("has_spouse_binary")

    df_cd["has_spouse_binary"] =  df_cd.filter(like='spouse').notna().any(axis=1).astype(int)
    columns_to_drop = ["spouse_id", "spouse_name", "spouse_business_name", \
                    "spouse_business_title", "spouse_email"]
    df_cd = df_cd.drop(columns=columns_to_drop)


    ### **Date to Age in Decimal**
    ##############################

    if verbose==1:
        print("Date to Age in Decimal")

    columns_age = ["age", "spouse_age"]
    for col in columns_age:
        df_cd[col] = pd.to_datetime(df_cd[col])
        dob_array = df_cd[col]
        ages_decimal = np.array(["%2.2f" %((today - dob).days/365.25) for dob in dob_array]).astype(float)
        ages_decimal[ages_decimal < 0] = np.nan
        df_cd = df_cd.drop(columns=[col])
        df_cd[col] = ages_decimal

    ### **Age Binning**
    ###################

    if verbose==1:
        print("Age binning")

    col = "age"
    bin_edges = [0, 30, 40, 50, 200]
    A = pd.cut(df_cd[col], bins=bin_edges, labels=False, right=False)

    # Convert binned ages into binary representation
    binary_ages = pd.get_dummies(A, prefix=col)

    # Join the binary columns to the original DataFrame
    df_cd = pd.concat([df_cd, binary_ages], axis=1)

    # Rename the binary columns
    binary_column_names = ["%s_%s_binary" %(col, edge) for edge in bin_edges[1:]]
    column_mapping = {binary_ages.keys().to_list()[i]:binary_column_names[i] for i in range(len(binary_ages.keys()))}
    df_cd.rename(columns=column_mapping, inplace=True)

    ### **Dates to Days**
    #####################

    if verbose==1:
        print("Dates to Days")

    column_dates = [key for key in df_cd.keys() if "date" in key.lower()] + ["class_year"]
    for col in column_dates:
        df_cd[col] = pd.to_datetime(df_cd[col])
        col_days = (today - df_cd[col]).dt.days
        col_days[col_days < 0] = np.nan
        df_cd = df_cd.drop(columns=col)
        df_cd[col] = col_days

    ### **Prefix**
    ##############

    if verbose==1:
        print("Checking if Prefix has Dr. or Prof.")

    df_cd["prefix"] = df_cd["prefix"].astype(str)
    df_cd["prefix_has_dr_binary"] = df_cd["prefix"].str.contains(r"(dr|prof)", case=False).astype(int)
    df_cd = df_cd.drop(columns=["prefix"])

    ### **Incomplete address**
    ##########################

    if verbose==1:
        print("Incomplete Address")

    df_cd["incomplete_address_binary"] = ((df_cd['address_1'].isna()) | (df_cd['home_city'].isna()) | \
                                (df_cd['home_state'].isna()) | (df_cd['zip'].astype(str).str.len() < 5)).astype(int)

    ### **Presence/Absence**
    ########################

    if verbose==1:
        print("Presence/Absence in columns to binary")

    columns_binaries = ["middle_name", "address_2", "head_of_household", "number_of_children", \
                        "history_of_volunteer", "employer_name", "business_address", \
                        "seasonal_address", "business_email", \
                        "home_phone", "cell_phone", "business_phone"]
    # Convert non-null entries into binary columns
    binary_df = pd.get_dummies(df_cd[columns_binaries].notnull().astype(int))
    column_mapping = {key:"%s_binary" %key for key in binary_df.keys()}
    binary_df.rename(columns=column_mapping, inplace=True)

    # Drop the original columns
    df_cd = df_cd.drop(columns=columns_binaries)

    # Concatenate the binary columns with the original DataFrame
    df_cd = pd.concat([df_cd, binary_df], axis=1)

    ### **TOP 5 BINARIES**
    ######################

    if verbose==1:
        print("Top-5 Binaries")

    # # Any Columns with Email
    # col_emails = [key for key in df_cd.keys() if "email" in key.lower()]
    col_emails = ["personal_email"]
    for col in col_emails:
        df_cd[col] = df_cd[col].astype(str)
        if len(df_cd[df_cd[col].notna()]) & len(df_cd[df_cd[col]!="nan"])>0:
            col_type = df_cd[col].str.split(pat="@", expand=True)[1]
            df_cd = df_cd.drop(columns=col)
            df_cd[col] = col_type
            df_cd = melt_ntop(df=df_cd, ntop=5, column=col)

    # %%
    columns = ["home_city", "home_state", "suffix", "last_action_type", "marital_status",\
            "constituent_type_1", "constituent_type_2", "number_of_special_events_attended"]
    for col in columns:
        df_cd = melt_ntop(df=df_cd, ntop=5, column=col)

    ### **Filters and Indicators**
    ##############################

    df_cd = df_cd.drop(columns=["is_deceased", "is_individual"])
    df_cd = df_cd[(df_cd["deceased"].str.contains("no", case=False)) & \
                (df_cd["key_indicator"].str.contains("I", case=False)) &\
                (df_cd["home_country"].str.contains("USA|U\.S\.A\.|United States|America", case=False)) ]
    df_indicators = pd.concat([df_cd.pop(col) for col in ["deceased", "key_indicator"]], axis=1)

    ### **Taggers**
    ###############

    column_taggers = ["first_name", "last_name", "home_country", "address_1", "zip",\
                    "current_trustee", "past_trustee", "assigned_manager", \
                    "lifetime_hard_credits", "lifetime_soft_credits", \
                    "first_gift_amount", "most_recent_gift_amount", "number_of_gifts"]
    df_taggers = pd.concat([df_cd.pop(col) for col in column_taggers], axis=1)
    df_taggers["constituent_id"] = df_cd["constituent_id"]

    ### **Sklearn -- preprocessing**
    ################################

    if verbose==1:
        print("Sklearn -- preprocessing")

    ### Drop all columns that are NaN
    df_reg = df_cd.dropna(how="all", axis=1)

    ### Drop all columns where 80% of the entires are NaN
    df_reg = df_reg.dropna(axis=1, thresh=len(df_reg)*0.20)

    constituent_id = df_reg.pop("constituent_id")

    ### Convert the lifetime giving into log1p
    df_reg["lifetime_giving"] = np.log1p(df_reg["lifetime_giving"])/np.log(10)

    columns_binary = [key for key in df_reg.keys() if "binary" in key.lower()]
    if len(columns_binary)>0:
        df_reg_subset = pd.concat([df_reg.pop(col) for col in columns_binary], axis=1)
    else:
        df_reg_subset = pd.DataFrame()
    df_reg_subset["m_giving_logp1"] = df_reg.pop("lifetime_giving")

    X = df_reg

    ### Preprocessing the predictors (scaling numeric variables and encoding categorical variables, feature_selection etc)
    numeric_features = [col for col in X.columns if is_numeric_dtype(X[col])]
    categorical_features = [col for col in X.columns if not is_numeric_dtype(X[col])]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehotencoder', OneHotEncoder(min_frequency=0.05))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    X_processed = preprocessor.fit_transform(X)

    ### Reset indices of X_processed and df_reg_subset
    df_reg_subset = df_reg_subset.reset_index(drop=True)

    final = pd.concat([pd.DataFrame(data=X_processed, columns=X.columns), \
                                df_reg_subset], axis=1)
    final = final.dropna(how="any")
    m_giving_logp1 = final.pop('m_giving_logp1')

    ### **Feature Selection**
    #########################

    if verbose==1:
        print("Feature Selection")

    ### y
    y = m_giving_logp1
    ### All columns
    xcol_all = final.keys().tolist()

    ### **Variance Threshold**
    X = final[xcol_all]
    for threshold in [0.01, 0.05, 0.1]:
        selector = VarianceThreshold(threshold=threshold)
        X_reduced = selector.fit_transform(X, y)

        cols = selector.get_support(indices=True)
        ncols = len(cols)
        globals() ["xcol_var_%d" %(threshold*100)] = X.iloc[:,cols].columns.tolist()

    ### **F_Statistic Threshold**
    q = 0.01
    dfn = 1
    dfd = len(X) - 2
    f01 = sp.stats.f.isf(q, dfn, dfd)

    # %%
    X = final[xcol_all]
    f_stat, p_values = f_regression(X, y)
    cols_f_stat = list(np.where(f_stat>f01)[0])
    xcol_f_stat = X.iloc[:,cols_f_stat].columns.tolist()

    ### **Pearson_R Threshold**
    X = final[xcol_all]
    r_pearson = np.abs(r_regression(X, y))
    cols = list(np.where(r_pearson>=np.mean(r_pearson))[0])
    ncols = len(cols)
    xcol_rpearson_mean = X.iloc[:,cols].columns.tolist()
    len(xcol_rpearson_mean)

    ### **Recursive Feature Elimination**
    #####################################

    if verbose==1:
        print("Recursive Feature Elimination")

    est_dict = {"lm": linear_model.LinearRegression(), "xgb": xgb.XGBRegressor(booster="gbtree"), \
                "xgbrf": xgb.XGBRFRegressor(), "svr": SVR(kernel="linear"), \
                "rf": RandomForestRegressor(n_estimators=100, oob_score=True, bootstrap=True, random_state=42)
            }

    run = True 
    if run:
        for est in ["xgb", "lm"]:
            estimator = est_dict[est]
            X = final[xcol_all]
            y = m_giving_logp1
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
            rfe = RFECV(estimator=estimator, step=1, min_features_to_select=1, verbose=0, \
                        n_jobs=(multiprocessing.cpu_count()//2), cv=3,\
                        scoring="neg_mean_squared_error")
            #print("Fitting %s" %est)
            rfe.fit(X=X_train, y=y_train)
            #print(f"Optimal number of features: {rfe.n_features_}")

            cols = rfe.get_support(indices=True)
            globals()["xcol_rfecv_%s" %est] = X.iloc[:,cols].columns.tolist()
            print("Estimator:", est, "No_of_features:", len(cols))

    ### **Model Fitting**
    #####################

    if verbose==1:
        print("Model Fitting")

    run = True
    if run:
        folds = 5

        ### X
        features = {"all": xcol_all, "var_1": xcol_var_1, "var_10": xcol_var_10, "var_5": xcol_var_5, \
                    "f_stat": xcol_f_stat, "r_pearson": xcol_rpearson_mean,\
                    "rfecv_xgb": xcol_rfecv_xgb, "rfecv_lm": xcol_rfecv_lm}
                    # "f_stat_med": xcol_f_stat_med,

        ### List of models to be tested
        algorithms = {"LR": linear_model.LinearRegression(), "GBR": ensemble.GradientBoostingRegressor(), \
                    "XGBR": xgb.XGBRegressor(), "XGBRF": xgb.XGBRFRegressor(),\
                    "DTR": tree.DecisionTreeRegressor()}

        # features = {"rfecv_xgb": xcol_rfecv_xgb, "rfecv_lm": xcol_rfecv_lm}
        # algorithms = {"XGBR": xgb.XGBRegressor(), "LR": linear_model.LinearRegression()}

        model_feature_importance = {}
        model_pred = {}
        model_stats = {}
        model_evaluation = {}

        y = m_giving_logp1
        for feature_type, cols in features.items():

            if verbose==1:
                print(feature_type)
            
            X = final[cols]
            
            # Creating a training set index by partitioning the dataset
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
            #print("feature_type = %10s | No. of cols = %3d" %(feature_type, len(cols)))
        
            model_feature_importance[feature_type] = {}
            model_pred[feature_type] = {}
            model_stats[feature_type] = {}
            model_evaluation[feature_type] = {}
            
            globals() ["model_df_%s" %(feature_type)] = pd.DataFrame({"Features": X.keys().tolist()})
            for algo_name, model in algorithms.items():
                
                ### Fitting
                scores = cross_val_score(model, X_train, y_train, cv=folds, scoring='neg_root_mean_squared_error')
                results = model.fit(X_train, y_train)
                y_pred = results.predict(X_test)
            
                ### Statistics
                # Calculate MAE, MSE and RMSE
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
            
                ### Populate the dictionaries and the dataframe

                try:
                    if hasattr(model, 'feature_importances_'):
                        model_feature_importance[feature_type][algo_name] = results.feature_importances_
                    elif hasattr(model, 'coef_'):
                        model_feature_importance[feature_type][algo_name] = results.coef_.flatten()
                #except AttributeError:
                #    print("%s object has no attribute feature_importances_ or coef_" %algo_name)
                except:
                    pass

                model_pred[feature_type][algo_name] = y_pred
                model_stats[feature_type][algo_name] = [mae, mse, rmse]
                model_evaluation[feature_type][algo_name] = -scores.mean()
                if verbose==2:
                    print(feature_type, algo_name)
                #, X_train.shape, len(cols), model_feature_importance[feature_type][algo_name])
                
                try:
                    globals() ["model_df_%s" %(feature_type)]["Coeff_%s" %algo_name] = model.coef_
                except:
                    pass  

        # Your list of tuples
        data = [(key, len(features[key]), key1, model_stats[key][key1][2]) for key in model_stats.keys() for key1 in model_stats[key].keys()]

        # Create a DataFrame
        df_matrix = pd.DataFrame(data, columns=["Feature_Selection", "N_Features", "Estimator", "Root_Mean_Square_Error"])
        df_matrix.to_csv("%s_pred_mod_matrix.csv", index=True)

    ### Best Fit Model
    ##################
    print("Fitting for the Best-Fit Model")
    best_fit_idx = df_matrix.Root_Mean_Square_Error.idxmin()
    best_fit_algo_name = df_matrix.iloc[best_fit_idx]["Estimator"]
    best_fit_feature_type = df_matrix.iloc[best_fit_idx]["Feature_Selection"]

    model = algorithms[best_fit_algo_name]
    cols = features[best_fit_feature_type]
    X = final[cols]
    y = m_giving_logp1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)
    results = model.fit(X_train, y_train)
    y_pred_train = results.predict(X_train)
    y_pred_test = results.predict(X_test)
    y_pred_all = results.predict(X)
    y_scores = y_pred_all*(100.0/y_pred_all.max())

    final["constituent_id"] = constituent_id
    final["m_giving_1p"] = y
    final["scores"] = y_scores
    df_final = final[["constituent_id", "m_giving_1p", "scores"]].merge(df_taggers, on=["constituent_id"], how="left")
    df_final.to_csv("%s_pred_mod_scores.csv" %client, index=False)

    # save
    save=False
    if save:
        with open("bestfit_results_%s_%s.pkl" %(best_fit_algo_name, best_fit_feature_type),"wb") as f:
            pickle.dump(results,f)
        with open("bestfit_model_%s_%s.pkl" %(best_fit_algo_name, best_fit_feature_type),"wb") as f:
            pickle.dump(model,f)


    ### Diagnostic Plots for the Best-Fit Model
    ###########################################

    if verbose==1:
        print("Diagnostic Plots")

    # Ground Truth vs Predicted
    colors = cm.prism(np.linspace(0, 1, len(model_pred.keys())))
    plt.figure(figsize=(4, 4))
    plt.scatter(y, y_pred_all, color='green', alpha=0.5, s=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.savefig("%s_pred_mod_scatter_plot.jpg" %client, bbox_inches="tight")

    # Residuals
    residuals = y_test - y_pred_test
    plt.figure(figsize=(4, 4))
    plt.hist(residuals, bins=20)
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.title('Distribution of Residuals')
    plt.savefig("%s_pred_mod_residuals_hist.jpg" %client, bbox_inches="tight")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictive Modeling based on Constituents File.")
#    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2], default=0, \
        help="Set verbosity level (0: no verbose, 1: medium, 2: high)")
    parser.add_argument("-c", "--columns-to-drop", nargs='+', default=DEFAULT_COLUMNS_TO_DROP, \
        help="Columns to drop from the dataframe (default: %(default)s)")
    args = parser.parse_args()
    main(verbose=args.verbose)


