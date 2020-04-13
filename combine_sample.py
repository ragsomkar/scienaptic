"""
This script is going to do following tasks,
1. Combine the data sets
2. Randomly sample the data into dev and val (80:20) sets.
3. Generate sample statistics
4. Generae descriptive statistics for categorical and numerical variables
5. Do one hot encoding for non binary categorical variables

"""
import pandas as pd
import numpy as np
import random

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def sampling(data, unique_id, split=0.2, random_state=545454):
    """
    Randomly sample the data usig input unique_id.
    """
    dev_val = data.copy()
    random.seed(random_state)
    dev_users = random.sample(list(dev_val[unique_id].unique()), int(
        round(dev_val[unique_id].nunique()*(1-split))))

    mask = dev_val[unique_id].isin(dev_users)
    dev_df = dev_val.loc[mask]
    dev_df['sample'] = 'dev'
    val_df = dev_val.loc[~mask]
    val_df['sample'] = 'val'

    final = pd.concat([dev_df, val_df], axis=0)
    final.reset_index(drop=True, inplace=True)

    return final

def num_missing(x):
    return np.sum(pd.isnull(x))


def num_unique(x):
    return x.nunique()


def num_zero(x):
    return len(x[x == 0])


def get_summary_stat(df, dv, cols):
    """
     Get descriptive statistics for input numerical cols 
    """
    data = df.copy()
    print ('# observations:', data.shape[0])
    summary1 = pd.DataFrame({'features': cols})
    summary1['dtype'] = pd.Series(data[cols].dtypes.values)
    summary1['min'] = pd.Series(data[cols].apply(np.nanmin, axis=0).values)
    summary1['max'] = pd.Series(data[cols].apply(np.nanmax, axis=0).values)
    summary1['mean'] = pd.Series(data[cols].apply(np.nanmean, axis=0).values)
    summary1['median'] = pd.Series(data[cols].apply(
     np.nanmedian, axis=0).values)
    summary1['#Missing'] = pd.Series(data[cols].apply(
     num_missing, axis=0).values)
    summary1['PercentMissing'] = summary1['#Missing'].apply(
     lambda x: x/float(len(data)))
    summary1['#Unique'] = pd.Series(data[cols].apply(
     num_unique, axis=0).values)
    summary1['#Zero'] = pd.Series(data[cols].apply(num_zero, axis=0).values)
    summary1['std'] = pd.Series(data[cols].apply(np.nanstd, axis=0).values)

    dv_corr_dict = {}
    for var in dv:
        dv_corr_dict[var]=[]
        
    P1 = []
    P25 = []
    P50 = []
    P75 = []
    P80 = []
    P85 = []
    P90 = []
    P95 = []
    P99 = []
    for i in cols:
        P1.append(np.nanpercentile(data[i], 1))
        P25.append(np.nanpercentile(data[i], 25))
        P50.append(np.nanpercentile(data[i], 50))
        P75.append(np.nanpercentile(data[i], 75))
        P80.append(np.nanpercentile(data[i], 80))
        P85.append(np.nanpercentile(data[i], 85))
        P90.append(np.nanpercentile(data[i], 90))
        P95.append(np.nanpercentile(data[i], 95))
        P99.append(np.nanpercentile(data[i], 99))
        mask = data[i].isnull()
        for var in dv:
            dv_corr_dict[var].append(np.corrcoef(data[~mask][i], data[~mask][var])[0][1])
        
    summary1['P1'] = pd.Series(P1)
    summary1['P25'] = pd.Series(P25)
    summary1['P50'] = pd.Series(P50)
    summary1['P75'] = pd.Series(P75)
    summary1['P80'] = pd.Series(P80)
    summary1['P85'] = pd.Series(P85)
    summary1['P90'] = pd.Series(P90)
    summary1['P95'] = pd.Series(P95)
    summary1['P99'] = pd.Series(P99)
    for var in dv:
        summary1['Corr_'+var] = pd.Series(dv_corr_dict[var])
    return summary1

def get_cat_variable_dist(data, cols):
    """
     Get basic statistics for input categorical cols 
    """
    dist_list = []
    for col in cols:
        print(col)
        dist = pd.DataFrame(data.groupby([col]).agg({'prospectid':np.size, '30_dpd_f3m':np.mean, 'bounce_f3m':np.mean}))
        dist.reset_index(inplace=True)
        dist['feature'] = col
        dist.set_index('feature', inplace=True)
        dist.rename(columns={col:'values'},inplace=True)
        dist_list.append(dist)
    final = pd.concat(dist_list)    
    return final

def add_dummy_vars(inp_data, unique_id, cols):
    """
     Create one-hot encoded dummy variables for input categorical cols
    """
    data = inp_data.copy()
    cat_feat_df_list = []
    for col in cols:
        print(col)
        data[col] = data[col].apply(lambda x: x.strip().replace(" ", "_").lower())
        df = pd.get_dummies(data[col])
        df.columns = [col+"_"+c for c in df.columns]
        cat_feat_df_list.append(df)

    cat_feat_df = pd.concat(cat_feat_df_list, axis=1)
    cat_feat_df[unique_id] = data[unique_id]
    return cat_feat_df

if __name__=="__main__":

    path = "/Users/raghavendramo/Project/Scienaptic/"

    data1=pd.read_csv(path+"rawdata/case_study1.csv")
    data2=pd.read_csv(path+"rawdata/case_study2.csv")
    
    #Combine data
    data=pd.merge(data1, data2, on=['PROSPECTID'], how='left')
    data.columns = [col.lower() for col in data.columns]
    data = sampling(data, 'prospectid')

    #Get Sample Statistics
    print(data.groupby(['sample']).agg({'prospectid':np.size, '30_dpd_f3m':np.mean, 'bounce_f3m':np.mean}))
    
    #Create non feature columns list
    unique_id = ['prospectid']
    dv_cols = ['30_dpd_f3m', 'bounce_f3m']
    sample_col = ['sample']
    non_feat_cols = unique_id + dv_cols + sample_col
    
    #Consider special values as missing values
    for col in data.columns:
        data.loc[(data[col] == -99999.0), col] = None
    
    #Create feature columns list
    feat_cols = list(set(data.columns) - set(non_feat_cols))
    
    #Identify binary and non-binary vcategorical variables
    cat_vars = [col for col in feat_cols if (data[col].dtype == 'object') or (data[col].nunique() == 2)]
    bin_cat_vars = [col for col in cat_vars if data[col].nunique() == 2]
    non_bin_cat_vars = set(cat_vars) - set(bin_cat_vars)

    # Get descriptive statistics for continous variables
    cont_vars = list(set(feat_cols) - set(cat_vars))
    cont_vars_summary = get_summary_stat(data, dv_cols, cont_vars)
    cont_vars_summary.to_csv(path + "Summary_Statisctics_Cont_Variables.csv", index=0)

    # Get basic statistics for categorical variables
    cat_vars_summary = get_cat_variable_dist(data, cat_vars)
    cat_vars_summary.to_csv(path + "Summary_Statisctics_Cat_Variables.csv")
    
    # Convert binary categorical variables having non binary values into binary values(1,0) 
    for col in bin_cat_vars:
        if (data[col].dtype == 'object'):
           print(col) 
           data[col].replace(dict(zip(data[col].unique(), [1,0])), inplace=True)

    # Add one hot encoded variables for categorical variables
    cat_feat_df = add_dummy_vars(data, unique_id[0], non_bin_cat_vars) 
    
    # Combine all features inro one dataset.
    final_data = pd.merge(data[cont_vars + bin_cat_vars + non_feat_cols], cat_feat_df, on=unique_id, how='left')
    final_feat_cols = list(set(final_data.columns) - set(non_feat_cols))
    for col in final_feat_cols:
        final_data[col] = final_data[col].astype(float)

    final_data.to_csv(path+"preprocess/case_study_combined_sampled.csv",index=0)    


