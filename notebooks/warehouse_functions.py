# the functions below simulate the production of many samples the length of the data
# then produces replicates (statistics) for each sample (in this case, mean)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from imblearn.over_sampling import SMOTE as SMOTE_imb
from imblearn.pipeline import Pipeline as imb
from sklearn.metrics import roc_curve
# generating a bootstrap replicate
def bootstrap_rep(data, func):
    return func(np.random.choice(data, size=len(data)))

# generating many bootstrap replicates
def draw_bs_reps(data, func, size=1):
    '''simulate generation of a statistic many times'''
    #Initialize array of replicates
    bs_reps = []

    #generate replicates
    for i in range(size):
        rep = bootstrap_rep(data, func)
        bs_reps.append(rep)
    return bs_reps

#defining the set of features we want to run hypothesis tests on
df = pd.read_csv('retail_warehouses.csv', index_col=0)
hypothesis_cols = df.drop(['county','state','state_abbrev','amazon_indicator','walmart_indicator','warehouse_target'], axis=1)

# divide the dataset in two: one set with counties where a warehouse is present, the other warehouse is absent
present = df.loc[df['warehouse_target']==1, :]
absent = df.loc[df['warehouse_target']==0, :]

#list of 1000 replicates for each feature
means = []
for feature in hypothesis_cols:
    wh_present = present[feature].values
    wh_absent = absent[feature].values

    means_present = draw_bs_reps(wh_present, np.mean, 1000)
    means_absent = draw_bs_reps(wh_absent, np.mean, 1000)
    mean_set = (means_present, means_absent)
    means.append(mean_set)

def hist_plot_1():
    index_cols = list(hypothesis_cols.columns)

    fig, ax = plt.subplots(5, 4, figsize=(18, 18))
    for i, var in enumerate(index_cols): 
        row = i//4
        col = i%4
        ax[row][col].hist(means[i][0])
        ax[row][col].hist(means[i][1])
        ax[row][col].title.set_text(var)
    
    return fig, axs



