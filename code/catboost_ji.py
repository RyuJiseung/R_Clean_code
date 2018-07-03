#!/usr/bin/env python

# This is a port to CatBoost + undersampling of 
# Alexey Pronin's tranlation of Krishna's LGBM into Python. 
# Big thanks to Andy Harless for his Pranav's 
# LGBM Python code -- I used it as a starter code. 
# Big thank you to Pranav and all others who contributed!
#############################################################################
FILENO = 5 #To distinguish the output file name.
debug=0  #Whethere or not in debuging mode
import os
import random
from distutils.dir_util import mkpath
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split 
import catboost as cb
import gc
import operator
from imblearn.under_sampling import RandomUnderSampler 
##############################################################################
out_suffix = '_CB_1_1'

TEST = False
TEST_SIZE = 100000
VALIDATE = True
MAX_ROUNDS = 1500
EARLY_STOP = 100
OPT_ROUNDS = 1200

USED_RAM_LIMIT = 5*(2 ** 30)

CLASSES_RATIO = 1.0
##############################################################################
total_rows = 184903890
##############################################################################
if TEST:
    rows_train = TEST_SIZE
    rows_test = TEST_SIZE
    skip_train = None
    comp_suffix = '.test'
else:
    rows_train = 40000000
    rows_test = None 
    skip_train = range(1, total_rows - rows_train + 1)
    comp_suffix = ''
##############################################################################
input_path = '/home/rjs/바탕화면/adtrack/data/'

path_train = input_path + 'train.csv'
path_test = input_path + 'test.csv'


intermed_path = './intermed' + comp_suffix + '/'

output_path = '/home/rjs/바탕화면/adtrack/result/'

mkpath(intermed_path)
mkpath(output_path)


##############################################################################
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
##############################################################################

##############################################################################
gc.collect()
##############################################################################
print('Building features...')
most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
least_freq_hours_in_test_data = [6, 11, 15]


def do_next_Click( df,agg_suffix='nextClick', agg_type='float32'):
    
    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    # {'groupby': ['ip', 'channel']},
    # {'groupby': ['ip', 'os']},
    
    # V3
    {'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    {'groupby': ['ip', 'os', 'device']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - df.click_time).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)

def do_prev_Click( df,agg_suffix='prevClick', agg_type='float32'):

    print(f">> \nExtracting {agg_suffix} time calculation features...\n")
    
    GROUP_BY_NEXT_CLICKS = [
    
    # V1
    # {'groupby': ['ip']},
    # {'groupby': ['ip', 'app']},
    {'groupby': ['ip', 'channel']},
    {'groupby': ['ip', 'os']},
    
    # V3
    #{'groupby': ['ip', 'app', 'device', 'os', 'channel']},
    #{'groupby': ['ip', 'os', 'device']},
    #{'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),agg_suffix)    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']

        # Run calculation
        print(f">> Grouping by {spec['groupby']}, and saving time to {agg_suffix} in: {new_feature}")
        df[new_feature] = (df.click_time - df[all_features].groupby(spec[
                'groupby']).click_time.shift(+1) ).dt.seconds.astype(agg_type)
        
        predictors.append(new_feature)
        gc.collect()
    return (df)    




## Below a function is written to extract count feature by aggregating different cols
def do_count( df, group_cols, agg_type='uint16', show_max=False, show_agg=True ):
    agg_name='{}count'.format('_'.join(group_cols))  
    if show_agg:
        print( "\nAggregating by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
    
##  Below a function is written to extract unique count feature from different cols
def do_countuniq( df, group_cols, counted, agg_type='uint8', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_countuniq'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCounting unqiue ", counted, " by ", group_cols ,  '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract cumulative count feature  from different cols    
def do_cumcount( df, group_cols, counted,agg_type='uint16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_cumcount'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCumulative count by ", group_cols , '... and saved in', agg_name  )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
### Below a function is written to extract mean feature  from different cols
def do_mean( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_mean'.format(('_'.join(group_cols)),(counted))  
    if show_agg:
        print( "\nCalculating mean of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_type='float16', show_max=False, show_agg=True ):
    agg_name= '{}_by_{}_var'.format(('_'.join(group_cols)),(counted)) 
    if show_agg:
        print( "\nCalculating variance of ", counted, " by ", group_cols , '... and saved in', agg_name )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    predictors.append(agg_name)
#     print('predictors',predictors)
    gc.collect()
    return( df )
    
##############################################################################
def prep_data( df ):
    ##############################################################################
    print('hour')
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    print('day')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    ##############################################################################
    print('in_test_hh')
    df['in_test_hh'] = (   2 
                         - 1*df['hour'].isin(  most_freq_hours_in_test_data ) 
                         + 1*df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_day_test_hh')                                                        
    gp = df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
            'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_day_test_hh'})
    df = df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    df.drop(['in_test_hh'], axis=1, inplace=True)
    df['n_ip_day_test_hh'] = df['n_ip_day_test_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_os_day_hh')
    gp = df[['ip', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'os', 'day',
            'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_os_day_hh'})
    df = df.merge(gp, on=['ip','os', 'day', 'hour'], how='left')
    del gp
    df['n_ip_os_day_hh'] = df['n_ip_os_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_day_hh')
    gp = df[['ip', 'app', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'day',
            'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_day_hh'})
    df = df.merge(gp, on=['ip', 'app', 'day', 'hour'], how='left')
    del gp
    df['n_ip_app_day_hh'] = df['n_ip_app_day_hh'].astype('uint16')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_os_day_hh')
    gp = df[['ip', 'app', 'os', 'day', 'hour', 'channel']].groupby(by=['ip', 'app', 'os', 
            'day', 'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_os_day_hh'})
    df = df.merge(gp, on=['ip', 'app', 'os', 'day', 'hour'], how='left')
    del gp
    df['n_ip_app_os_day_hh'] = df['n_ip_app_os_day_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_app_day_hh')
    gp = df[['app', 'day', 'hour', 'channel']].groupby(by=['app', 
            'day', 'hour'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_app_day_hh'})
    df = df.merge(gp, on=['app', 'day', 'hour'], how='left')
    del gp
    df['n_app_day_hh'] = df['n_app_day_hh'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df.drop( ['day'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    ##############################################################################
    print('group by : n_ip_app_dev_os')
    gp = df[['ip', 'app', 'device', 'os', 'channel']].groupby(by=['ip', 'app', 
             'device', 'os'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_app_dev_os'})
    df = df.merge(gp, on=['ip', 'app', 'device', 'os'], how='left')
    del gp
    df['n_ip_app_dev_os'] = df['n_ip_app_dev_os'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df['ip_app_dev_os_cumcount'] = df.groupby(['ip', 'app', \
                                                'device', 'os']).cumcount().astype('uint16')
    ##############################################################################
    print('group by : n_ip_dev_os')
    gp = df[['ip', 'device', 'os', 'channel']].groupby(by=['ip', 'device', 
             'os'])[['channel']].count().reset_index().rename(index=str, 
            columns={'channel': 'n_ip_dev_os'})
    df = df.merge(gp, on=['ip', 'device', 'os'], how='left')
    del gp
    df['n_ip_dev_os'] = df['n_ip_dev_os'].astype('uint32')
    gc.collect()
    print( df.info() )
    ##############################################################################
    df['ip_dev_os_cumcount'] = df.groupby(['ip', 'device', 'os']).cumcount().astype('uint16')
    ##############################################################################
    df.drop( ['ip'], axis=1, inplace=True )
    gc.collect()
    print( df.info() )
    ##############################################################################
    return( df )
    

#### A function is written here to run the full calculation with defined parameters.
nrows=184903891-1
nchunk=40000000
val_size=4000000
frm=nrows-nchunk
#frm = 1
to=frm+nchunk

#def DO(frm,to,FILENO):
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint8',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

print('loading train data...',frm,to)
train_df = pd.read_csv("/home/rjs/바탕화면/adtrack/data/train.csv", parse_dates=['click_time'], 
                       skiprows=range(1,frm), nrows=to-frm, 
                       dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
if debug:
    test_df = pd.read_csv("/home/rjs/바탕화면/adtrack/data/test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
else:
    test_df = pd.read_csv("/home/rjs/바탕화면/adtrack/data/test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df=train_df.append(test_df)
    
del test_df

gc.collect()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('int8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('int8') 
train_df = do_next_Click( train_df,agg_suffix='nextClick', agg_type='float32'  ); gc.collect()
train_df = do_prev_Click( train_df,agg_suffix='prevClick', agg_type='float32'  ); gc.collect()  ## Removed temporarily due RAM sortage. 

train_df = do_countuniq( train_df, ['ip'], 'channel' ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour' ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app'); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os'); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device'); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel'); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os'); gc.collect()
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app'); gc.collect()
train_df = do_count( train_df, ['ip', 'day', 'hour'] ); gc.collect()
train_df = do_count( train_df, ['ip', 'channel']); gc.collect()
train_df = do_count( train_df, ['ip', 'app']); gc.collect()
train_df = do_count( train_df, ['ip', 'app','channel'] ); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os']); gc.collect()
train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour'); gc.collect()
train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour'); gc.collect()
train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day'); gc.collect()
train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour' ); gc.collect()
train_df = do_mean( train_df, ['ip', 'app','os'], 'hour' ); gc.collect()
train_df = do_mean( train_df, ['ip', 'channel'], 'hour' ); gc.collect()
train_df = prep_data(train_df)

train = train_df
del train_df
gc.collect()
gc.collect()

test = train[len_train:]
train = train[:len_train]
##############################################################################
print( "Train info before processing: ")
train.info()
##############################################################################
#train = prep_data( train )
gc.collect()
##############################################################################
print("Variables and data type: ")
train.info()
##############################################################################
eval_metric = 'AUC'
##############################################################################
cb_params = {
    # technical
    'verbose' : True,
    'random_seed' : 42,
    'save_snapshot' : True,
    'snapshot_file' : output_path + 'snapshot',
    'used_ram_limit' : USED_RAM_LIMIT,
    
    # learning
    'l2_leaf_reg' : 150,
    'scale_pos_weight' : CLASSES_RATIO,
    'one_hot_max_size' : 100,
    'max_ctr_complexity' : 3,
    'leaf_estimation_iterations' : 8,
    'learning_rate' : 0.1,
    'eval_metric' : eval_metric
}
##############################################################################
target = 'is_attributed'
##############################################################################
predictors = list(set(train.columns) - {'is_attributed', 'click_time'})
print('The list of predictors:')
for item in predictors: print(item)
##############################################################################
categorical = ['app', 'device', 'os', 'channel', 'hour']

categorical_features_indices = [predictors.index(cat_name) for cat_name in categorical]

print('categorical_features_indices', categorical_features_indices)
##############################################################################
print(train.head(5))
##############################################################################
alltrain_X = train[predictors].values
alltrain_Y = train[target].values


if VALIDATE:
    ##############################################################################
    
    train_X, val_X, train_Y, val_Y = train_test_split(
        alltrain_X, alltrain_Y, train_size=.95, random_state=99, shuffle=True,
        stratify = alltrain_Y
    )
    ##############################################################################
    print("train shape: ", train_X.shape)
    print("valid shape: ", val_X.shape)
    ##############################################################################
    gc.collect()

    ##############################################################################
    print("Undersampling...")
    ##############################################################################
    target_counts = np.bincount(train_Y)
    print('target_counts', target_counts)
    
    rus = RandomUnderSampler(random_state=42, 
                             ratio={0: int(CLASSES_RATIO*target_counts[1]),
                                    1: target_counts[1]})
    
    #uns_train_X, uns_train_Y = rus.fit_sample(train_X, train_Y)
    uns_train_X, uns_train_Y = train_X, train_Y
    
    target_counts = np.bincount(train_Y)
    print('target_counts after undersamping', target_counts)
    ##############################################################################
    print("Training...")
    ##############################################################################
    cb_params["iterations"] = MAX_ROUNDS
    cb_params["od_type"] = 'Iter'
    cb_params["od_wait"] = EARLY_STOP
        
    dtrain = cb.Pool(uns_train_X,
                     label=uns_train_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )
    
    dvalid = cb.Pool(val_X,
                     label=val_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )    
    del uns_train_X
    del uns_train_Y
    del val_X
    del val_Y
    gc.collect()
    ##############################################################################
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(dtrain, eval_set=dvalid)
    cb_model.save_model(output_path + "Krishna_s_train_uns_w_valid" + out_suffix + ".cbm")
    
    del dvalid
    
else:
    ##############################################################################
    print(train.info())
    ##############################################################################
    print("train size: ", len(train))
    ##############################################################################
    gc.collect()
    ##############################################################################
    print("Training...")
    ##############################################################################
    cb_params['iterations']=OPT_ROUNDS
    ##############################################################################
    dtrain = cb.Pool(alltrain_X, label=alltrain_Y,
                     feature_names=predictors,
                     cat_features=categorical_features_indices
                    )
    del alltrain_X
    del alltrain_Y
    gc.collect()
    ##############################################################################
    cb_model = cb.CatBoostClassifier(**cb_params)
    cb_model.fit(dtrain)
    cb_model.save_model(output_path + "Krishna_s_alltrain_uns_" + out_suffix + ".cbm")
    ##############################################################################

del dtrain
gc.collect()

print('Model params')
print(cb_model.get_params())

##############################################################################
print('Loading test...')
#test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
#test = pd.read_csv(path_test, nrows=rows_test, dtype=dtypes, usecols=test_cols)
##############################################################################
#test = prep_data( test )
gc.collect()
##############################################################################
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
##############################################################################
print("Predicting...")
pred_probs = cb_model.predict_proba(test[predictors].values)
sub['is_attributed'] = [prob[1] for prob in pred_probs]
##############################################################################
print("Writing prediction to a csv file...")
sub.to_csv(output_path + 'Krishna_s_CatBoost_1_1' + out_suffix + '.csv',
           index=False)
print(sub.info())
##############################################################################
print("All done!..")
##############################################################################
