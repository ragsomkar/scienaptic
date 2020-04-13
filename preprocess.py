"""
This script is going to preprocess the rawdata by applying folllowing data prpeprocessing steps.
1.Imputer : Impute missing values of each variable using their median value 
2.Outlierscaler : Treat outlier values by capping any value above (P75 + (1.5 * IQR(P75 - P25))) and below (P25 - (1.5 * IQR(P75 - P25))).
3.Stdscaler : Standardize the values using Z scores.
"""
import numpy as np
import pandas as pd
import pickle
import random

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

def calc_outlier_bounds(numbers):
    """Return low and high bounds which excludes outliers."""

    l, h = np.percentile(numbers, [25, 75])
    r = (h - l) * 1.5
    return (max(l - r, np.min(numbers)),
            min(h + r, np.max(numbers)))

class LegacyOutlierScaler(BaseEstimator, TransformerMixin):
    """ Scales outliers based on IQR """
    def __init__(self, copy=True):

        self.copy = copy

    def fit(self, X, y=None):

        self.low = []
        self.high = []
        self.low_default = []
        self.high_default = []
        self.low_scaler = []
        self.high_scaler = []

        for j in range(X.shape[1]):
            x = X[:, j].copy()
            low, high = calc_outlier_bounds(x)
            self.low.append(low)
            self.high.append(high)
            if low == high:
                self.low_default.append(None)
                self.low_scaler.append(None)
                self.high_scaler.append(None)
                self.high_default.append(None)
                continue
            q1, q3 = np.percentile(x, [25, 75])
            qr = (q3 - q1) / 2.
            qrd = qr / 50

            min_, max_ = x.min(), x.max()

            if min_ < low:
                self.low_scaler.append(min(qr / (low - min_), 1))
                self.low_default.append(None)
            else:
                self.low_scaler.append(None)
                self.low_default.append(low - qrd)

            if max_ > high:
                self.high_scaler.append(min(qr / (max_ - high), 1))
                self.high_default.append(None)
            else:
                self.high_scaler.append(None)
                self.high_default.append(high + qrd)
        return self

    def transform(self, X, y=None, copy=None):

        if self.copy:
            X_ = X.copy()
        else:
            X_ = X
        for j in range(X.shape[1]):
            
            low = self.low[j]
            mask = X_[:, j] < low
            if any(mask):
                default = self.low_default[j]
                scaler = self.low_scaler[j]
                if default is not None:
                    X_[mask, j] = default
                elif scaler is not None:
                    X_[mask, j] = low - \
                        (low - X_[mask, j]) * self.low_scaler[j]
                    
            high = self.high[j]
            mask = X_[:, j] > high
            if any(mask):
                default = self.high_default[j]
                scaler = self.high_scaler[j]
                if default is not None:
                    X_[mask, j] = default
                elif scaler is not None:
                    X_[mask, j] = high + \
                        (X_[mask, j] - high) * self.high_scaler[j]
        return X_


if __name__=="__main__":
    path = "/Users/raghavendramo/Project/Scienaptic/"
    data=pd.read_csv(path+"preprocess/case_study_combined_sampled.csv")

    unique_id = ['prospectid']
    dv_cols = ['30_dpd_f3m', 'bounce_f3m']
    sample_col = ['sample']
    non_feat_cols = unique_id + dv_cols + sample_col

    feat_cols = list(set(data.columns) - set(non_feat_cols))
    feat_cols.sort()

    dev = data[data['sample']== 'dev']
    val = data[data['sample'] == 'val']
    DV = '30_dpd_f3m'
    X = dev[feat_cols].values
    y = dev[DV].values

    steps = [('imputer', SimpleImputer(missing_values=np.nan, strategy='median')),
             ('outlierscaler', LegacyOutlierScaler()),
             ('stdscaler', StandardScaler())]

    pipeline = Pipeline(steps)   
    pipeline.fit(X, y)
    prep_data = pd.DataFrame(pipeline.transform(data[feat_cols].values), columns=feat_cols)
    prep_data = pd.concat([prep_data, data[non_feat_cols]], axis=1)
    prep_data.to_csv(path+"preprocess/case_study_combined_sampled_preprocessed.csv", index=0)
    pickle.dump(pipeline, open(path+"preprocess/case_study_combined_sampled_preprocessed.pkl", 'wb'))


 
