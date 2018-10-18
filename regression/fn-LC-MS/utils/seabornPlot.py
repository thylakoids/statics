import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plotcorrelationMatrix(d: pd.DataFrame):
    # d pandas.DateFrame
    sns.set(style="white")
    # Compute the correlation matrix
    corr = d.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    # plt.show()
    f.savefig('correlation.pdf', bbox_inches='tight')


def plotfeatureImportance(d: pd.DataFrame, title: str):
    # d: dataframe, first column:x , second column: y
    sns.set(style='white')
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(18, 8))
    columns = d.columns
    sns.barplot(x=columns[0], y=columns[1], data=d)
    ax.set_title(title)
    # plt.show()
    f.savefig("{}.pdf".format(title), bbox_inches='tight')
