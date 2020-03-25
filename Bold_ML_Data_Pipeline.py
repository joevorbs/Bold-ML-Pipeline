

#################################################
#Machine Learning Data Pipeline for Bold Data 
#Work in Progress
#################################################

#Read in packages
from google.cloud import bigquery
from google.oauth2 import service_account
from multiprocessing import Pool
import multiprocessing
import json
import pandas as pd
import numpy as np
import re
import h2o
from h2o.targetencoder import TargetEncoder
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.grid.grid_search import H2OGridSearch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:20.2f}'.format

#Initialize h2o
h2o.init(max_mem_size = 10)


#############
#READ IN DATA
#############

# Sling credentials to big query
credentials = service_account.Credentials.from_service_account_file(
    '/Users/joevorbeck/desktop/key2.json')

project_id = 'thankful-68f22'

client = bigquery.Client(credentials= credentials,project=project_id)

#Pull everything from the app's table
query_job = client.query("""
 SELECT *
 FROM `thankful-68f22.analytics_198686154.events_*`""")
results = query_job.result()


#Sling results to a dataframe
df = results.to_dataframe()


######################
#SET GLOBAL VARIABLES
#####################


#Global Variables - can be changed
max_one_hot = 25       #Threshold for categorical data
min_numerical = 300    #Threshold for numerical data
amt_to_sample = 100    #Numer of rows to sample
split_ratio = .25      #Ratio for train/hold
dep_var =  "engage_ind"          #Target for model - example case
scaling_method = MinMaxScaler()  #Method for Scaling (MinMax, Standard, Etc.)
seed = 1994


##########
#CREATE DV
##########


#Create DV - example case
df[dep_var] = df.apply(lambda row: 1 if row['event_name'] == 'user_engagement' else 0, axis = 1)


#########################################
#TURN COLS WITH DICTS INTO SEPARATE COLS
#########################################


#Multiple columns have dictionaries as rows - convert each key to it's own column
cleaned_geo = pd.DataFrame(df['geo'].values.tolist())
cleaned_device = pd.DataFrame(df['device'].values.tolist())
cleaned_app_info = pd.DataFrame(df['app_info'].values.tolist())

#Append the new columns to the raw data
df_cleaned_1 = pd.concat((df, cleaned_geo, cleaned_device, cleaned_app_info), axis = 1)

#Drop old columns that still contain the dicts - also dropping event params, traffic source, and user properties
#These are nested dictionaries within arrays and will be cleaned later, traffic source also only has 1 value
df_cleaned_1 = df_cleaned_1.drop(['geo','device','app_info','event_params', 'user_id',
                                  'traffic_source','user_properties', 'user_pseudo_id', 'vendor_id'],axis = 1)


################
#IDENTIFY DATES
################


#Check for datetime fields
datetimes_list = list()

for col in df_cleaned_1.columns:  #Loop through columns
    try:
        col_to_check = str(df_cleaned_1[col].sample(amt_to_sample).dropna().iloc[0]) #Sample 100 non-na values from a column and check first values
        date_match = re.findall(r'^[0-9]{8}$', col_to_check) #Regex match a 4 digit year - 2 month - 2 day pattern
        timestamp_match = re.findall(r'^[0-9]{16}$', col_to_check)
        
        if date_match or timestamp_match:   #If match is found, place the name of the column in an empty list
            datetimes_list.append(col)            
            
    except:   #If not continue - try/except is here as some columns are entirely null and the loop will go out of range on the iloc
            continue


#Create dataframe of just dates
#Add column from original dataframe, no need to calculate event time differences
df_dates = df_cleaned_1[datetimes_list]
df_dates['event_previous_timestamp'] = df['event_previous_timestamp']

#Create column for time between last event and previous
df_dates['event_time_diff'] = df_dates['event_timestamp'] - df_dates['event_previous_timestamp']
#Impute 0 for nulls in the case that there isnt a previous event to compare to
df_dates['event_time_diff'] = df_dates['event_time_diff'].fillna(0)

####################################
#DETECT CARDINALITY OF OTHER COLUMNS
####################################


#Check cardinality of other fields for proper treatment
low_card = list()
high_card = list()
numerical = list()
single_value = list()
all_na = list()

for col in df_cleaned_1.drop(datetimes_list, axis=1):
    try:
        col_count_dist = df_cleaned_1[col].nunique()
        if col_count_dist == 0:
            all_na.append(col)
        elif col_count_dist == 1:
            single_value.append(col)
        elif col_count_dist <= max_one_hot:
            low_card.append(col)
        elif col_count_dist > max_one_hot and col_count_dist <= min_numerical:
            high_card.append(col)
        elif col_count_dist > min_numerical:
            numerical.append(col)
    except:
        pass

#Create dataframes by cardinality
df_low_card = df_cleaned_1[low_card]
df_high_card = df_cleaned_1[high_card]
df_numerical = df_cleaned_1[numerical]

#####################
#IDENTIFY BINARY COLS
#####################


#Find columns that are already binary - reduces redundancy in one hot encoding
already_binary = list()

for col in df_low_card.columns:
    try:
        col_min = df_low_card[col].min()
        col_max = df_low_card[col].max()
        if col_min == 0 and col_max == 1:
            already_binary.append(col)
    except:
        pass

#Dataframe of already binary columns
already_binary.remove(dep_var)
df_binary = df_low_card[[already_binary]]


#############################
#ONE HOT ENCODE LOW CARD DATA
#############################


#Drop already binary cols from df_low_card and one hot encode
df_low_one_hot = df_low_card.drop(already_binary, axis = 1)
df_low_one_hot = pd.get_dummies(df_low_card).drop(dep_var, axis = 1)


#################################
#REGEX NUMERICAL COLS TO FIND 
#CATEGORICAL DATA THAT LEAKED
#THEN SCALE & NORMALIZE TRUE NUMS
#################################


#Regex match on numerical data to see what is truly numerical or not
true_num = list()
true_high_card = list()

for col in df_numerical.columns:
    try:
        col_to_check = str(df_numerical[col].sample(amt_to_sample).dropna().iloc[0])
        num_match = re.match("^[0-9]+$", col_to_check)
        if num_match:
            true_num.append(col)
        else:
            true_high_card.append(col)
    except:
        pass

#Dataframe of truly numerical data - drop values that were categorical
#Convert to h2o frame for faster mean imputation
df_true_numerical = h2o.H2OFrame(df_numerical.drop(true_high_card, axis = 1))


#Impute missing values with column mean
for i in df_true_numerical.columns:
    df_true_numerical.impute(i, method="mean")


#Convert back to dataframe
df_true_numerical = df_true_numerical.as_data_frame()

#Apply scaler
df_true_numerical_scaled = scaling_method.fit(df_true_numerical).transform(df_true_numerical)
df_true_numerical_scaled = pd.DataFrame(df_true_numerical_scaled, columns = df_true_numerical.columns)

########################################
#CREATE DATAFRAME OF TRUE HIGH CARD DATA
#AND TARGET ENCODE
########################################                              


#Add high cardinality categorical data to the high card list and create a df
high_card.extend(true_high_card)


#Dataframe of true high cardinality data
df_true_high_card = df_cleaned_1[high_card]


#Add target column for encoding
df_true_high_card[dep_var] = df[dep_var]


#Create train/test datasets for encoding
te_train, te_test = train_test_split(df_true_high_card, test_size = split_ratio, random_state = seed)


#Turn train and test set into h2o frames
te_train = h2o.H2OFrame(te_train).ascharacter().asfactor()
te_test = h2o.H2OFrame(te_test).ascharacter().asfactor()

#Create a fold column
fold = te_train.kfold_column(n_folds=5, seed=1234)
fold.set_names(["fold"])

te_train = te_train.cbind(fold)
te_train["fold"] = te_train["fold"].asfactor()

#Set list of columns to encode - drop DV from variable list
x = list(te_train.columns)
x = [i for i in x if i not in [dep_var, "fold"]]

#Initialize target encoder and fit to training set
target_encoder = TargetEncoder(x = x, y = dep_var,
                               fold_column="fold",
                               blended_avg= True,
                               inflection_point = 3,
                               smoothing = 1,
                               seed=1234)
target_encoder.fit(te_train)


#Transform training set
encoded_train = target_encoder.transform(frame=te_train, holdout_type="kfold", noise=0.2, seed=1664)

#Transform test set
encoded_test = target_encoder.transform(frame=te_test, holdout_type="none", noise=0.0, seed=1)

#Union two sets backtogther and convert back to df
df_high_card_te = encoded_train.drop("fold", axis = 1).rbind(encoded_test)
df_high_card_te = df_high_card_te.as_data_frame()

#Drop untransformed columns
df_high_card_te.drop(x, axis = 1, inplace=True)

##########################################
#MERGE ALL TRANSFORMED DATAFRAMES TOGETHER
##########################################                            

dates_to_binary = pd.concat([df_dates, df_binary], axis = 1)

dates_binary_to_low_card = pd.concat([dates_to_binary, df_low_one_hot], axis = 1)

dates_binary_low_to_num = pd.concat([dates_binary_to_low_card, df_true_numerical_scaled], axis = 1)

all_to_high_card = pd.concat([dates_binary_low_to_num, df_high_card_te], axis = 1)

###############
#TEST MODELLING
###############                         

#Turn final dataset into an h2o frame
df_converted = h2o.H2OFrame(all_to_high_card)

#Create train/holdout sets
df_test, df_train = df_converted.split_frame(ratios = [split_ratio], seed = seed)

#Isolate IVs - drop event_name as its target leakage (just for example here)
features = df_test.columns
del features[-4]

#Lasso Regression
glm = H2OGeneralizedLinearEstimator(family= "binomial", lambda_ = .00015, alpha = 1)
glm.train(features, dep_var, training_frame = df_train)

#Confusion Matrix
glm.confusion_matrix()

#AUC
glm.auc()

#Test set performance
preds = glm.predict(df_test)
glm.model_performance(df_test)

#Close H2O
h2o.cluster().shutdown()

