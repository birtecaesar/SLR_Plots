import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
import itertools
import seaborn as sns


def clean_data(filepath):
    data = pd.read_csv(filepath, sep=";", header=None)

    for idx_row, row_content in data.iterrows():
        for idx_col, content in row_content.iteritems():
            if isinstance(content, str):
                if idx_row == 0:
                    if content.startswith("# "):
                        data.loc[idx_row, idx_col] = int(content[2:4])
                    elif content.startswith("#"):
                        data.loc[idx_row, idx_col] = int(content[1:3])
                elif idx_row == 1:
                    if content.startswith("0,"):
                        data.loc[idx_row, idx_col] = float("0." + content[2:])
                elif idx_row == 18:
                    if content.startswith("10"):
                        data.loc[idx_row, idx_col] = int(content[:2])
                    else:
                        data.loc[idx_row, idx_col] = int(content[:1])
                else:
                    try:
                        data.loc[idx_row, idx_col] = int(content[:1])
                    except ValueError:
                        pass
    return data

def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)

# main #
df_e = pd.DataFrame(columns=["Component", "Usage"])
# load and clean analysis data
data = clean_data(filepath="./2022-11-07_DTF_Evaluation.csv")

data2plot = data.drop(index=[0, 1]).dropna()

# count and sort data for plots
data2plot_single = data2plot.apply(pd.Series.value_counts, axis=1)

## Process Planing
usage = data2plot_single.loc[10].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[0] = pd.Series({"Component": "Process Planing", "Usage": usage})

## Layout Design
usage = data2plot_single.loc[9].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[1] = pd.Series({"Component": "Layout Design", "Usage": usage})

## Path Planing
usage = data2plot_single.loc[8].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[2] = pd.Series({"Component": "Path Planing", "Usage": usage})

## Scheduling
usage = data2plot_single.loc[11].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[3] = pd.Series({"Component": "Scheduling", "Usage": usage})

## Capability Manager
usage = data2plot_single.loc[4].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[4] = pd.Series({"Component": "Capability Manager", "Usage": usage})

## Interoperability Manager
usage = data2plot_single.loc[21].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[5] = pd.Series({"Component": "Interoperability Manager", "Usage": usage})

## Data Collection
usage = data2plot_single.loc[23].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[6] = pd.Series({"Component": "Data Collection", "Usage": usage})

## Data Processing
usage = data2plot_single.loc[22].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[7] = pd.Series({"Component": "Data Processing", "Usage": usage})

## Configuration Knowledge
usage = data2plot_single.loc[24].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[8] = pd.Series({"Component": "Configuration Knowledge", "Usage": usage})

## Decision Maker
usage = data2plot_single.loc[17].dropna()
usage = usage.drop(index=[0, 1]).sum()
df_e.loc[9] = pd.Series({"Component": "Decision Maker", "Usage": usage})

## Reconfiguration Trigger Analysis
usage = data2plot_single.loc[2].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[10] = pd.Series({"Component": "Reconfiguration Trigger Analysis", "Usage": usage})

## Eastbound Interface
usage = data2plot_single.loc[17].dropna()
usage = usage.drop(index=[0, 2]).sum()
df_e.loc[11] = pd.Series({"Component": "Eastbound Interface", "Usage": usage})

## Configuration Evaluation
usage = data2plot_single.loc[13].dropna()
usage = usage.drop(index=[0]).sum()
df_e.loc[12] = pd.Series({"Component": "Configuration Evaluation", "Usage": usage})

## Add Usage Percentage
df_e["Usage_rel"] = df_e["Usage"] / 67


# create bar plot
fig, ax = plt.subplots()
ax2 = ax.twiny()
sns.set_theme(style="whitegrid")
ax = sns.barplot(data=df_e, x="Usage", y="Component", ax=ax, palette=colors_from_values(df_e["Usage"].astype(float), "Greens_d"))#palette="dark:#5A9_r", ax=ax)
ax2 = sns.barplot(data=df_e, x="Usage_rel", y="Component", alpha=0.0, ax=ax2)
# ax.bar_label(ax.containers[0])
ax.set(ylabel='')
ax2.set(xlabel='Usage in %')
plt.tight_layout()
# plt.show()
plt.savefig("EvaluationResult.pdf")