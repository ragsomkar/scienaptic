"""
This script is going to do following tasks
1. Do RandomizedSearchCV with GBC algo for both dependent variables('30_dpd_f3m', 'bounce_f3m')
2. Find the best model by calculating overfit of auc metric between train and test sample.
3. Rebuild the best model and generate necessary model arifacts such as model object, scores, and feature importance.
4. Do univariate analysis on predominant features of both models
"""
import ast
import numpy as np
import pandas as pd
import pickle
import matplotlib.pylab as pl

from collections import Counter
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix, log_loss

   
def tier_cutoffs(probas, ntiers=10):
    """
    Return tier cuts calculated based on equal width binning.
    """
    p = probas.copy()
    p.sort()
    c = np.unique(p[np.linspace(0, len(p)-1, ntiers + 1, dtype=int)])
    if len(c) == 1:
        c = np.array([-np.inf, c[0]])
    if len(c) == 2:
        return np.array([[v,v] for v in c])   
    return np.array(list(zip(c[:-1], c[1:])))


def calc_tiers(probas, cutoffs=None, ntiers=10):
    """
    Return tiers calculated based on equal width binning.
    """
    tiers = np.zeros(len(probas), int)

    if cutoffs is None:
        cutoffs = tier_cutoffs(probas, ntiers)

    cutoffs = cutoffs.astype(float)
    cutoffs[0, 0] = -np.inf
    cutoffs[-1, -1] = np.inf
    for t, (l, h) in enumerate(cutoffs, start=1):
        tiers[(l < probas) & (probas <= h)] = t

    return tiers


def tier_table(target, probas, cutoffs, tiers=None, ntiers=10):
    """
     Calculate cumulative gains table based on the input cutoffs
    """
    if tiers is None:
        tiers = calc_tiers(probas, cutoffs, ntiers)

    tc = Counter(tiers)
    index = list(tc)
    index.sort()
    size = np.array([tc[i] for i in index])
    tt = pd.DataFrame(size, columns=['Size'], index=index)
    tt['C_Size'] = size.cumsum()
    tt['Bad'] = np.array([target[tiers == t].sum() for t in index])
    nbads = target.sum()
    ngoods = len(target) - nbads

    tt['C_Bad'] = tt['Bad'].cumsum()
    tt['BadRate'] = (tt['Bad'] / tt['Size'])
    tt['C_BadRate'] = (tt['Bad'].cumsum() / tt['Size'].cumsum())

    tt['%Good'] = ((tt['C_Size'] - tt['C_Bad']) / ngoods)
    tt['%Bad'] = (tt['C_Bad'] / nbads)
    tt['KS'] = (tt['%Good'] - tt['%Bad']) * 100.
    tt['Mean'] = np.array([probas[tiers == t].mean()for t in index])
    tt['Min'] = np.array([probas[tiers == t].min()for t in index])
    tt['Max'] = np.array([probas[tiers == t].max()for t in index])
    return tt

def make_lists(target, probas):
    """
    Make target and probas lists
    """
    if isinstance(probas, list):
        if not isinstance(target, list):
            target = [target] * len(probas)
    else:
        probas = [probas]
        target = [target]

    return target, probas


def get_class_1_probas(probas):
    """Return probabilities for class 1 when probabilities for both classes
    are given, otherwise return probabilities
    """
    if probas.ndim > 1:
        probas = probas[:, 1]
    if probas.dtype != float:
        return probas.astype(float)
    else:
        return probas


def plot_roc_curves(target, probas, dv, s_names=None):
    """
    Plot roc curve(s)
    """
    target, probas = make_lists(target, probas)

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    for i, (t, p) in enumerate(zip(target, probas), start=1):
        p = get_class_1_probas(p)
        fpr, tpr, thresholds = roc_curve(t, p)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if s_names is not None:
           i = s_names[i-1]
        pl.plot(fpr, tpr, lw=1,
                label='ROC fold %s (area = %0.2f)' % (str(i), roc_auc))

    mean_tpr /= len(probas)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    pl.plot(mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))

    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('ROC Curve(s) {}'.format(dv))
    pl.legend(loc="lower right")
    pl.show()

def get_goodbad_cumul_rates(t, p):
    """
     Returns cumulative distribution of goods and bads
    """
    zipped = list(zip(p, t, np.abs(1 - t)))
    zipped.sort()
    zipped = np.array(zipped)
    nb, ng = zipped[:, 1:].sum(0)
    assert nb + ng == len(p)
    br = zipped[:, 1].cumsum() * 100. / nb
    gr = zipped[:, 2].cumsum() * 100. / ng
    return gr, br

def calc_ks(target, probas):
    """
    Returns KS which is the maximum separation between cumulative distribution of goods and bads
    """
    gr, br = get_goodbad_cumul_rates(target, probas)
    ks = (gr - br).max()
    return ks

def plot_ks_curves(target, probas, dv, s_name, **kwargs):
    """Plot roc curve(s)."""

    import matplotlib.pylab as pl
    target, probas = make_lists(target, probas)
    markers = ['-', '--', '-.', ':', '+', 'x']
    for i, (t, p, m) in enumerate(zip(target, probas, markers), start=1):
        if s_name is not None:
           i = s_name[i-1]
        p = get_class_1_probas(p)
        gr, br = get_goodbad_cumul_rates(t, p)
        x = np.arange(len(p)) * 100. / len(p)
        pl.plot(x, gr, 'g' + m, **kwargs)
        pl.plot(x, br, 'r' + m, **kwargs)
        ks = (gr - br)
        index = ks.argmax()
        pl.plot([x[index], x[index]], [br[index], gr[index]], 'b' + m,
                label='KS %s = %0.2f' % (str(i), ks[index]) )

    pl.xlim([-5, 105])
    pl.ylim([-5, 105])
    pl.xlabel('Cumulative Dist Bad')
    pl.ylabel('Cumulative Dist Good')
    pl.title('KS Curve(s) {}'.format(dv))
    pl.legend(loc="upper left")
    pl.show()


def get_model_results(data, feat_cols, dv, cv_res_path, overfit_threshold=0.05, n_model=25):
    """
    1. Select best models by applying overfit_threshold on overfit calculated between train and test sample
    2. Rebuild n best models from selection and record performance metrics such as AUC and KS for each of them.
    """
    dev = data[data['sample']== 'dev']
    val = data[data['sample'] == 'val']

    X = dev[feat_cols].values
    y = dev[dv].values

    cv_res = pd.read_csv(cv_res_path)
    cv_res.rename(columns={"Unnamed: 0":'modelid'},inplace=True)

    cv_res['overfit'] = round((cv_res['mean_train_score'] - cv_res['mean_test_score'])/cv_res['mean_test_score'],3)
    cv_res_fil = cv_res[(cv_res['overfit'] > 0) & (cv_res['overfit'] <= overfit_threshold)]
    if cv_res_fil.shape[0] < n_model:
        n_model = cv_res_fil.shape[0] 
    models = random.sample(cv_res_fil['modelid'].to_list(), n_model)
    cv_res_fil_samp = cv_res_fil[cv_res_fil['modelid'].isin(models)]
    
    per_list = []
    for id in cv_res_fil_samp['modelid']:
        param = cv_res_fil_samp.loc[cv_res_fil_samp['modelid']==id, 'params'].values[0]
        print(param)
        model = GradientBoostingClassifier(**ast.literal_eval(param))
        model.fit(X, y)
        dev_scores = model.predict_proba(X)[:, 1]
        val_scores = model.predict_proba(val[feat_cols].values)[:, 1]
        dev_auc = roc_auc_score(dev[dv], dev_scores)
        val_auc = roc_auc_score(val[dv], val_scores)
        dev_val_overfit = (dev_auc - val_auc)/val_auc
        print("Dev AUC", dev_auc)
        print("Val AUC", val_auc)
        print("Dev Val Overfit", dev_val_overfit)
        per_list.append([id, dev_auc, val_auc, dev_val_overfit])

    per_df = pd.DataFrame(per_list, columns=['modelid', 'dev_auc','val_auc','dev_val_overfit'])   
    cv_res_fil_samp = pd.merge(cv_res_fil_samp, per_df, how='left', on=['modelid'])   
    return cv_res_fil_samp

def build_single_model(data, feat_cols, non_feat_cols, dv, cv_res_path, model_id):
    """
    1. Rebuild single model by selecting the model from crossval results with model_id.
    2. Generate model artifcats such as model object, scores, and feature importance.
    """
    score_df = data[non_feat_cols]

    dev = data[data['sample']== 'dev']
    X = dev[feat_cols].values
    y = dev[dv].values

    cv_res = pd.read_csv(cv_res_path)
    cv_res.rename(columns={"Unnamed: 0" : 'modelid'}, inplace=True)

    param = cv_res.loc[cv_res['modelid']==model_id, 'params'].values[0]
    print(param)
    model = GradientBoostingClassifier(**ast.literal_eval(param))
    model.fit(X, y)
    score_df['Score'] = model.predict_proba(data[feat_cols].values)[:, 1]
    print("Dev AUC", roc_auc_score(score_df[score_df['sample']=='dev'][dv], score_df[score_df['sample']=='dev']['Score']))
    print("Val AUC", roc_auc_score(score_df[score_df['sample']=='val'][dv], score_df[score_df['sample']=='val']['Score']))

    print("Dev KS", calc_ks(score_df[score_df['sample']=='dev'][dv], score_df[score_df['sample']=='dev']['Score']))
    print("Val KS", calc_ks(score_df[score_df['sample']=='val'][dv], score_df[score_df['sample']=='val']['Score']))

    print("Dev log loss", log_loss(score_df[score_df['sample']=='dev'][dv], score_df[score_df['sample']=='dev']['Score']))
    print("Val log loss", log_loss(score_df[score_df['sample']=='val'][dv], score_df[score_df['sample']=='val']['Score']))

    feat_df = pd.DataFrame({'Features': feat_cols, 'Importance': model.feature_importances_})
    mask = feat_df.Importance != 0
    feat_df = feat_df[mask]
    feat_df.sort_values('Importance', ascending=False, inplace=True)
    feat_df.reset_index(inplace=True, drop=True)

    return model, score_df, feat_df

def get_feat_dist(dev, val, dv, feat_imp_path, num_feats=5):
    """
    Generate dev/val gains tables for top n feats.
    """
    feat_df = pd.read_csv(feat_imp_path)
    res_list = []
    for feat in feat_df['Features'][:num_feats]:
        dev_cut_offs = tier_cutoffs(dev[feat].values)

        dev_tier_table = tier_table(dev[dv], dev[feat].values, dev_cut_offs)
        dev_tier_table.reset_index(inplace=True)
        dev_tier_table.rename(columns={'index':'Tiers'}, inplace=True)
        dev_tier_table['index'] = feat+"_dev"
        dev_tier_table.set_index('index', inplace=True)

        val_tier_table = tier_table(val[dv], val[feat].values, dev_cut_offs)
        val_tier_table.reset_index(inplace=True)
        val_tier_table.rename(columns={'index':'Tiers'}, inplace=True)
        val_tier_table['index'] = feat+"_val"
        val_tier_table.set_index('index', inplace=True)

        dev_val = pd.concat([dev_tier_table, val_tier_table])
        res_list.append(dev_val)

    final_feat_dist = pd.concat(res_list)
    return final_feat_dist

def get_precission_recall_metrics(data, dv, t=None):
    """
     Get Precission and Recall metrics for input probability threshold
    """
    score = data['Score']
    threshold_list = []
    acc_list = []
    precission_list = []
    recall_list = []
    f1_score_list = []
    if t is None:
       t_range = np.linspace(0, score.max(), 1000)
    else:
       t_range = [t] 
    for cut in t_range:
        mask = score > cut
        score_bin = np.zeros(len(score))
        score_bin[mask]=1
        tn, fp, fn, tp = confusion_matrix(data[dv], score_bin).ravel()
        threshold_list.append(cut)
        acc_list.append( (1.0 * (tp+tn)) / (tp+fn+fp+tn))
        p=(1.0 * tp)/ (tp+fp)
        precission_list.append(p)
        r = (1.0 * tp)/ (tp+fn)
        recall_list.append(r)
        f1_score_list.append(2 * ((p*r)/(p+r)))
    
    final = pd.DataFrame({"threshold":threshold_list, "acc":acc_list,"precission":precission_list,"recall":recall_list,"f1_score":f1_score_list})
    return final

if __name__=="__main__":
    path = "/Users/raghavendramo/Project/Scienaptic/"
    data=pd.read_csv(path+"preprocess/case_study_combined_sampled_preprocessed.csv")

    unique_id = ['prospectid']
    dv_cols = ['30_dpd_f3m', 'bounce_f3m']
    sample_col = ['sample']
    non_feat_cols = unique_id + dv_cols + sample_col

    feat_cols = list(set(data.columns) - set(non_feat_cols))
    feat_cols.sort()

    dev = data[data['sample'] == 'dev']
    val = data[data['sample'] == 'val']
    DV1 = '30_dpd_f3m'
    DV2 = 'bounce_f3m'
    X = dev[feat_cols].values
    y1 = dev[DV1].values
    y2 = dev[DV2].values
    
    #Hyper Parameter grid
    param_grid = {'max_depth':[2, 3],
                  'n_estimators':[250, 500, 750, 1000, 1500, 2000],
                  'learning_rate':[0.01, 0.03, 0.1],
                  'min_samples_leaf':[10, 30, 100],
                  'min_samples_split':[10, 30, 100],
                  'subsample':[0.7, 0.9],
                  'min_impurity_decrease' : [0, 0.01, 0.1],
                  'random_state':[54545454]
                  }

    #Do RandomizedSearchCV for DV (30_dpd_f3m)
    cv_res_path = path + "gridsearch/case_study_combined_sampled_preprocessed_gridsearch_result_30_dpd_f3m_v1.csv"
    gs = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=param_grid, n_iter=250,
                      scoring='roc_auc', n_jobs=10, cv=4, refit=False, return_train_score=True, verbose=20)
    gs_obj = gs.fit(X, y1)

    #Save cross-validation results
    pd.DataFrame(gs_obj.cv_results_).to_csv(cv_res_path)
    
    #Select best models
    best_models = get_model_results(data, feat_cols, DV1, cv_res_path)
    best_models.to_csv("{}_{}.csv".format(cv_res_path.split(".csv")[0],"best_models"), index=0)
    
    #Rebuild the best model and save model object, scores, and feature importance.
    best_model_id = 24
    model_obj, score_df, feat_df = build_single_model(data, feat_cols, non_feat_cols, DV1, cv_res_path, best_model_id)
    pickle.dump(model_obj, open(path+"gridsearch/model_"+DV1+".pkl", 'wb'))
    score_df.to_csv(path+"gridsearch/scores_"+DV1+".csv", index=0)
    feat_df.to_csv(path+"gridsearch/feature_importance_"+DV1+".csv", index=0)
    
    #Generate dev/val gains tables out of model score.
    score = pd.read_csv(path+"gridsearch/scores_"+DV1+".csv")
    dev_cut_offs = tier_cutoffs(score[score['sample']=='dev']['Score'].values)
    dev_tier_table = tier_table(score[score['sample']=='dev'][DV1], score[score['sample']=='dev']['Score'].values, dev_cut_offs)
    val_tier_table = tier_table(score[score['sample']=='val'][DV1], score[score['sample']=='val']['Score'].values, dev_cut_offs)
    print(dev_tier_table)
    print(val_tier_table)
    
    #Generate dev/val gains tables of top n features of model.
    feats_dist = get_feat_dist(dev, val, DV1, path+"gridsearch/feature_importance_"+DV1+".csv")
    feats_dist.to_csv(path+"gridsearch/feat_dist_"+DV1+".csv")

    #Plot AUC and KS curves
    dev_score = score[score['sample']=='dev']
    val_score = score[score['sample']=='val']
    plot_roc_curves([dev_score[DV1], val_score[DV1]], [dev_score['Score'], val_score['Score']], DV1, ['Dev', 'Val'])
    plot_ks_curves([dev_score[DV1], val_score[DV1]], [dev_score['Score'], val_score['Score']], DV1, ['Dev', 'Val'])


    #Do RandomizedSearchCV for DV (bounce_f3m)
    cv_res_path = path+"gridsearch/case_study_combined_sampled_preprocessed_gridsearch_result_bounce_f3m_v1.csv"
    gs = RandomizedSearchCV(estimator=GradientBoostingClassifier(), param_distributions=param_grid, n_iter=250,
                      scoring='roc_auc', n_jobs=10, cv=4, refit=False, return_train_score=True, verbose=20)
    gs_obj = gs.fit(X, y2)
    
    #Save cross-validation results
    pd.DataFrame(gs_obj.cv_results_).to_csv(cv_res_path)
    
    #Select best models
    best_models = get_model_results(data, feat_cols, DV2, cv_res_path)
    best_models.to_csv("{}_{}.csv".format(cv_res_path.split(".csv")[0],"best_models"), index=0)
    
    #Rebuild the best model and save model object, scores, and feature importance.
    bes_model_id = 18
    model_obj, score_df, feat_df = build_single_model(data, feat_cols, non_feat_cols, DV2, cv_res_path, bes_model_id)
    pickle.dump(model_obj, open(path+"gridsearch/model_"+DV2+".pkl", 'wb'))
    score_df.to_csv(path+"gridsearch/scores_"+DV2+".csv", index=0)
    feat_df.to_csv(path+"gridsearch/feature_importance_"+DV2+".csv", index=0)
   
    #Generate dev/val gains tables out of model score.
    score = pd.read_csv(path+"gridsearch/scores_"+DV2+".csv")
    dev_cut_offs = tier_cutoffs(score[score['sample']=='dev']['Score'].values)
    dev_tier_table = tier_table(score[score['sample']=='dev'][DV2], score[score['sample']=='dev']['Score'].values, dev_cut_offs)
    val_tier_table = tier_table(score[score['sample']=='val'][DV2], score[score['sample']=='val']['Score'].values, dev_cut_offs)
    print(dev_tier_table)
    print(val_tier_table)
    
    #Generate dev/val gains tables top n features of model.
    feats_dist = get_feat_dist(dev, val, DV2, path+"gridsearch/feature_importance_"+DV2+".csv")
    feats_dist.to_csv(path+"gridsearch/feat_dist_"+DV2+".csv")

    #Plot AUC and KS curves
    dev_score = score[score['sample']=='dev']
    val_score = score[score['sample']=='val']
    plot_roc_curves([dev_score[DV2], val_score[DV2]], [dev_score['Score'], val_score['Score']], DV2, ['Dev', 'Val'])
    plot_ks_curves([dev_score[DV2], val_score[DV2]], [dev_score['Score'], val_score['Score']], DV2, ['Dev', 'Val'])




   
p_r_metrics = get_precission_recall_metrics(dev, DV1)
p_r_metrics.to_csv(path+"gridsearch/p_r_metrics_"+DV1+".csv")

    