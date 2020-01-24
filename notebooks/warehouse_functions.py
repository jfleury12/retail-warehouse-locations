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

    columns = list(hypothesis_cols.columns)
    fig, axs = plt.subplots(5, 4,figsize=(15,15))

    axs[0, 0].hist(means[0][0])
    axs[0, 0].hist(means[0][1])
    axs[0, 0].set_title(f'{columns[0]}')
    axs[0, 1].hist(means[1][0])
    axs[0, 1].hist(means[1][1])
    axs[0, 1].set_title(f'{columns[1]}')
    axs[0, 2].hist(means[2][0])
    axs[0, 2].hist(means[2][1])
    axs[0, 2].set_title(f'{columns[2]}')
    axs[0, 3].hist(means[3][0])
    axs[0, 3].hist(means[3][1])
    axs[0, 3].set_title(f'{columns[3]}')
    axs[1, 0].hist(means[4][0])
    axs[1, 0].hist(means[4][1])
    axs[1, 0].set_title(f'{columns[4]}')
    axs[1, 1].hist(means[5][0])
    axs[1, 1].hist(means[5][1])
    axs[1, 1].set_title(f'{columns[5]}')
    axs[1, 2].hist(means[6][0])
    axs[1, 2].hist(means[6][1])
    axs[1, 2].set_title(f'{columns[6]}')
    axs[1, 3].hist(means[7][0])
    axs[1, 3].hist(means[7][1])
    axs[1, 3].set_title(f'{columns[7]}')
    axs[2, 0].hist(means[8][0])
    axs[2, 0].hist(means[8][1])
    axs[2, 0].set_title(f'{columns[8]}')
    axs[2, 1].hist(means[9][0])
    axs[2, 1].hist(means[9][1])
    axs[2, 1].set_title(f'{columns[9]}')
    axs[2, 2].hist(means[10][0])
    axs[2, 2].hist(means[10][1])
    axs[2, 2].set_title(f'{columns[10]}')
    axs[2, 3].hist(means[11][0])
    axs[2, 3].hist(means[11][1])
    axs[2, 3].set_title(f'{columns[11]}')
    axs[3, 0].hist(means[12][0])
    axs[3, 0].hist(means[12][1])
    axs[3, 0].set_title(f'{columns[12]}')
    axs[3, 1].hist(means[13][0])
    axs[3, 1].hist(means[13][1])
    axs[3, 1].set_title(f'{columns[13]}')
    axs[3, 2].hist(means[14][0])
    axs[3, 2].hist(means[14][1])
    axs[3, 2].set_title(f'{columns[14]}')
    axs[3, 3].hist(means[15][0])
    axs[3, 3].hist(means[15][1])
    axs[3, 3].set_title(f'{columns[15]}')
    axs[4, 0].hist(means[16][0])
    axs[4, 0].hist(means[16][1])
    axs[4, 0].set_title(f'{columns[16]}')
    axs[4, 1].hist(means[17][0])
    axs[4, 1].hist(means[17][1])
    # axs[4, 1].set_title(f'{columns[17]}')
    # axs[4, 2].hist(means[18][0])
    # axs[4, 2].hist(means[18][1])
    # axs[4, 2].set_title(f'{columns[18]}')
    # axs[4, 3].hist(means[19][0])
    # axs[4, 3].hist(means[19][1])
    # axs[4, 3].set_title(f'{columns[19]}')
    for ax in axs.flat:
        ax.set(xlabel='', ylabel='frequency')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    return fig, axs



