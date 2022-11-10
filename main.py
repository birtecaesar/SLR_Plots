import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict
import pandas as pd
import itertools
import worldmap
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-white")


def plot_pie_chart(vals, label, title, cm="BuGn"):
    fig, ax = plt.subplots()
    ax.pie(
        vals,
        labels=label,
        autopct="%1.0f%%",
        startangle=90,
        colors=plt.colormaps[cm](
            [int(240 - 190 * (len(vals) - x) / len(vals)) for x in np.arange(1, len(vals) + 1)]
        ),
    )
    # ax.set_title(title)
    plt.savefig(title + ".pdf", bbox_inches="tight", pad_inches=0)
    plt.close()


def count_combinations(idxs: List[int], df: pd.DataFrame) -> Dict[tuple, int]:
    # retrieve rows from dataframe
    rows = [df.loc[idx] for idx in idxs]
    # find unique entry options
    options = {idx_row: set(row.to_list()) for idx_row, row in enumerate(rows)}
    # initialize combination dict
    args = [v for v in options.values()]
    combinations = {comb: 0 for comb in set(itertools.product(*args))}
    # count combinations and update combination dict
    for idx_col in range(len(rows[0])):
        comb = tuple([row.iloc[idx_col] for row in rows])
        combinations[comb] += 1
    # remove zero entries from combinations
    combinations = {k: v for k, v in combinations.items() if v > 0}
    return combinations


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


def create_plot_121():
    global subgroup_size, subgroup_names, comb_count, group_size, group_names, a, b, c, d, e, hatches, fig, ax, mypie, _, mypie2, handles, labels, subgroup_names_legs, new_handles
    subgroup_size = [x for x in data2plot_single.loc[4].dropna()]
    subgroup_names = [f"{np.round(x * 100 / sum(subgroup_size),0)}%" for x in subgroup_size]
    comb_count = count_combinations([5, 6], data2plot)
    group_size = [
        comb_count[(0, 0)],
        comb_count[(1, 0)],
        comb_count[(0, 1)],
        comb_count[(1, 1)],
        comb_count[(2, 0)],
        comb_count[(0, 2)],
        comb_count[(2, 1)],
        comb_count[(1, 2)],
        comb_count[(2, 2)],
    ]
    group_names = [f"{np.round(x * 100 / sum(group_size),0)}%" for x in group_size]
    # Create colors
    a, b, c, d, e = [
        plt.cm.Greys,
        plt.cm.Blues,
        plt.cm.Greens,
        plt.cm.Oranges,
        plt.cm.Reds,
    ]
    hatches = ["", "-", "|", '+', "--", '||', '++', '+++', "", "", '', '']
    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis("equal")
    mypie, _ = ax.pie(
        group_size,
        radius=1.3,
        labels=group_names,
        labeldistance=0.78,
        colors=["white", b(0.3), b(0.5), b(0.7), c(0.3), c(0.4), c(0.5), c(0.6), "white"],
    )
    # for i in range(len(mypie)):
    #     mypie[i].set(hatch = hatches[i])
    plt.setp(mypie, width=0.5, edgecolor="white")
    # Second Ring (Inside)
    mypie2, _ = ax.pie(
        subgroup_size,
        radius=1.3 - 0.5,
        labels=subgroup_names,
        labeldistance=0.68,
        colors=[a(0.4), b(0.8), c(0.7), d(0.6), e(0.4)],
    )
    plt.setp(mypie2, width=0.5, edgecolor="white")
    plt.margins(-0.4, 5, tight=False)
    handles, labels = ax.get_legend_handles_labels()
    subgroup_names_legs = [
        "Not F",
        "HW PF",
        "SW PF",
        "HW & SW PF",
        "HW F & SW not F",
        "SW F & HW not F",
        "HW F & SW PF",
        "SW F & HW PF",
        "HW & SW F",
    ]
    new_handles = handles[9:10] + handles[1:8] + handles[12:13]
    ax.legend(new_handles, subgroup_names_legs, loc="center", bbox_to_anchor=(1.05, 0.25))
    # plt.legend(bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    plt.savefig("R121nestedPie.pdf")
    # plt.show()


def create_plot_122():
    global subgroup_size, subgroup_names, comb_count, group_size, group_names, a, b, c, d, e, hatches, fig, ax, mypie, _, mypie2, handles, labels, subgroup_names_legs, new_handles
    subgroup_size = [x for x in data2plot_single.loc[7].dropna()]
    subgroup_names = [f"{np.round(x * 100 / sum(subgroup_size), 0)}%" for x in subgroup_size]
    comb_count = count_combinations([8, 9, 10, 11], data2plot)
    group_size = [
        comb_count[(0, 0, 0, 0)],
        comb_count[(1, 0, 0, 0)],
        comb_count[(0, 0, 1, 0)],
        comb_count[(0, 0, 0, 1)],
        comb_count[(1, 1, 0, 0)],
        comb_count[(1, 0, 1, 0)],
        comb_count[(1, 0, 0, 1)],
        comb_count[(0, 0, 1, 1)],
        comb_count[(1, 1, 1, 0)],
        comb_count[(1, 0, 1, 1)],
        comb_count[(0, 1, 1, 1)],
        comb_count[(1, 1, 1, 1)],
    ]
    group_names = [f"{np.round(x * 100 / sum(group_size), 0)}%" for x in group_size]
    # Create colors
    a, b, c, d, e = [
        plt.cm.Greys,
        plt.cm.Blues,
        plt.cm.Greens,
        plt.cm.Oranges,
        plt.cm.Purples,
    ]
    hatches = ["", "-", "|", '+', "--", '||', '++', '+++', "", "", '', '']
    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis("equal")
    mypie, _ = ax.pie(
        group_size,
        radius=1.3,
        labels=group_names,
        labeldistance=0.78,
        colors=["white", b(0.3), b(0.5), b(0.7), c(0.3), c(0.4), c(0.5), c(0.6), d(0.3), d(0.5), d(0.7), "white"],
    )
    # for i in range(len(mypie)):
    #     mypie[i].set(hatch = hatches[i])
    plt.setp(mypie, width=0.5, edgecolor="white")
    # Second Ring (Inside)
    mypie2, _ = ax.pie(
        subgroup_size,
        radius=1.3 - 0.5,
        labels=subgroup_names,
        labeldistance=0.68,
        colors=[a(0.4), b(0.8), c(0.7), d(0.8), e(0.4)],
    )
    plt.setp(mypie2, width=0.5, edgecolor="white")
    plt.margins(tight=False)
    handles, labels = ax.get_legend_handles_labels()
    subgroup_names_legs = [
        "None fulfilled",
        "MS",  # "Machine selection",
        "PSA",  # "Production sequence adaptation",
        "S",  # "Scheduling",
        "MS & MP",  # "Machine selection & positioning",
        "MS & PSA",  # "Machine selection & production sequence adaptation",
        "MS & S",  # "Machine selection & scheduling",
        "PSA & S",  # "Production sequence adaptation & scheduling",
        "MS & MP & PSA",  # "Machine selection & positioning & production sequence adaptation",
        "MS & PSA & S",  # "Machine selection & production sequence adaptation & scheduling",
        "MP & PSA & S",  # "Machine positioning & production sequence adaptation & scheduling",
        "All fulfilled",
    ]
    new_handles = handles[12:13] + handles[1:11] + handles[16:17]
    ax.legend(new_handles, subgroup_names_legs, loc="center", bbox_to_anchor=(1.05, 0.32))
    # plt.legend(bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    plt.savefig("R122nestedPie.pdf")
    # plt.show()

def create_plot_13():
    global subgroup_size, subgroup_names, comb_count, group_size, group_names, a, b, c, d, e, hatches, fig, ax, mypie, _, mypie2, handles, labels, subgroup_names_legs, new_handles
    subgroup_size = [x for x in data2plot_single.loc[13].dropna()]
    subgroup_names = [f"{np.round(x * 100 / sum(subgroup_size), 1)}%" for x in subgroup_size]
    comb_count = count_combinations([14, 15, 16], data2plot)
    group_size = [
        comb_count[(0, 0, 0)],
        comb_count[(1, 0, 0)],
        comb_count[(0, 1, 0)],
        comb_count[(1, 1, 0)],
        comb_count[(1, 0, 1)],
        comb_count[(0, 1, 1)],
        comb_count[(1, 1, 1)],
    ]
    group_names = [f"{np.round(x * 100 / sum(group_size), 1)}%" for x in group_size]
    # Create colors
    a, b, c, d, e = [
        plt.cm.Greys,
        plt.cm.Blues,
        plt.cm.Greens,
        plt.cm.Oranges,
        plt.cm.Reds,
    ]
    hatches = ["", "-", "|", '+', "--", '||', '++', '+++', "", "", '', '']
    # First Ring (outside)
    fig, ax = plt.subplots()
    ax.axis("equal")
    mypie, _ = ax.pie(
        group_size,
        radius=1.3,
        labels=group_names,
        labeldistance=0.78,
        colors=["white", b(0.5), b(0.7), c(0.4), c(0.5), c(0.6), "white"],
    )
    # for i in range(len(mypie)):
    #     mypie[i].set(hatch = hatches[i])
    plt.setp(mypie, width=0.5, edgecolor="white")
    # Second Ring (Inside)
    mypie2, _ = ax.pie(
        subgroup_size,
        radius=1.3 - 0.5,
        labels=subgroup_names,
        labeldistance=0.68,
        colors=[a(0.4), b(0.8), c(0.7), d(0.6), e(0.4)],
    )
    plt.setp(mypie2, width=0.5, edgecolor="white")
    plt.margins(-0.4, 5, tight=False)
    handles, labels = ax.get_legend_handles_labels()
    subgroup_names_legs = [
        "None fulfilled",  # "Not considered",
        "RE",  # "Recon. effort considered",
        "PE",  # "Prod. effort considered",
        "RE & PE",  # "Recon. & Prod. effort considered",
        "RE & MCO",  # "Recon. effort & multi-criteria optimization considered",
        "PE & MCO",  # "Prod. effort and multi-criteria optimization considered",
        "All fulfilled",  # "All points considered",
    ]
    new_handles = handles[7:8] + handles[1:6] + handles[10:11]
    ax.legend(new_handles, subgroup_names_legs, loc="center", bbox_to_anchor=(0.96, 0.15))
    # plt.legend(bbox_to_anchor=(1.5, 1))
    plt.tight_layout()
    plt.savefig("R13nestedPie.pdf")
    # plt.show()

# main
# load and clean analysis data

# data = clean_data(filepath="./2022-10-25 R122.csv")
data = clean_data(filepath="./2022-10-25 Full_Text_RM_Survey_merged_cleaned.csv")

data2plot = data.drop(index=[0, 1]).dropna()

# count and sort data for pie plots
data2plot_single = data2plot.apply(pd.Series.value_counts, axis=1)

# data to latex table
data2tex = data.dropna()
data2tex = pd.DataFrame(data2tex.T)
data2tex = data2tex.sort_values(by=[0])
data2tex[1] = data2tex[1].map("{:.2%}".format)
data2tex = data2tex.set_axis(
    [
        "#",
        "R1",
        "R1.1",
        "R1.2",
        "R1.2.1",
        "R1.2.1.1",
        "R1.2.1.2",
        "R1.2.2",
        "R1.2.2.1",
        "R1.2.2.2",
        "R1.2.2.3",
        "R1.2.2.4",
        "R1.2.3",
        "R1.3",
        "R1.3.1",
        "R1.3.2",
        "R1.3.3",
        "R1.4",
        "R2",
        "R3",
        "R4",
        "R5",
    ],
    axis=1,
)

with open("Result_Table.tex", "w") as tab:
    tab.write(data2tex.to_latex(index=False))


# R2

plot_pie_chart(
    vals=data2plot_single.loc[18].dropna(),
    label=["1", "2", "3", "4", "5", "7", "8", "10"],
    title="R2LoA",
)


# R3
plot_pie_chart(
    vals=data2plot_single.loc[19].dropna(),
    label=["Not Fulfilled", "Partially Fulfilled", "Fulfilled"],
    title="R3CPPS",
)

# R4
plot_pie_chart(
    vals=data2plot_single.loc[20].dropna(),
    label=["No Support", "Manual Systematic Support", "Tool Supported", "Automated"],
    title="R4ModelCreation",
)

# R5
plot_pie_chart(
    vals=data2plot_single.loc[21].dropna(),
    label=["Not Fulfilled", "Partially Fulfilled", "Fulfilled"],
    title="R5Interoperability",
)

# R1
fig, ax = plt.subplots()
ax.pie(
    [20, 25, 19, 3],
    labels=["0%-25%\nRequirement fulfillment", "25%-50%\nRequirement fulfillment", "50%-75%\nRequirement fulfillment", "75%-100%\nRequirement fulfillment"],
    autopct="%1.1f%%",
    startangle=90,
    colors=plt.colormaps["BuGn"](
# [int(240 - 190 * (len(vals) - x) / len(vals)) for x in np.arange(1, len(vals) + 1)]
        [int(240 - 190 * (5-x) / 5) for x in np.arange(1, 5)]
    ),
)
# ax.set_title(title)
plt.savefig("R1Summary" + ".pdf", bbox_inches="tight", pad_inches=0)
plt.close()

#R1.1 call function plot_pie_charts
plot_pie_chart(
    vals=data2plot_single.loc[2].dropna(),
    label=["Not Fulfilled", "Partially Fulfilled", "Fulfilled"],
    title="R11IdentificationOfDemand",
)
# Further Insights
new_header = data.iloc[0] #grab the first row for the header
df = data[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header
# 66-70 only strategic trigger, 68-95 only operational trigger, 10-94 both
df = df[[66, 72, 44, 11, 55, 67, 35, 69, 48, 92, 81, 76, 70, 68, 15, 93, 71, 18, 56, 16, 95, 19, 10, 14, 62, 94]]
df = df.loc[[5, 6, 8, 9, 10, 11]]

## R1.2.1 call function plot_pie_charts
# plot_pie_chart(
#     vals=data2plot_single.loc[5].dropna(),
#     label=["Not Considered", "Partially Considered", "Fully Considered"],
#     title="R121HardwareConsideration",
# )

# plot_pie_chart(
#     vals=data2plot_single.loc[6].dropna(),
#     label=["Not Considered", "Partially Considered", "Fully Considered"],
#     title="R121SoftwareConsideration",
# )

## R1.2.1 nested pie chart -> needs to be cleaned up and generalized
# ['HW and SW are fully considered', 'HW or SW is fully considered','','HW and SW are partially considered', 'HW or SW is partially considered','','Non is considered',]
create_plot_121()

## R1.2.2 Nested Pie
# create_plot_122()

## R1.2.3 call function plot_pie_charts
plot_pie_chart(
    vals=data2plot_single.loc[12].dropna(),
    label=["Not Fulfilled", "Partially Fulfilled", "Fulfilled"],
    title="R123IntelligentSolutionSpace",
)

## R1.3 nested pie chart
# create_plot_13()

# Further Insights
new_header = data.iloc[0] #grab the first row for the header
df = data[1:] #take the data less the header row
df.columns = new_header #set the header row as the df header
# 17-30 only rec, 41-46 rec + 1, 48-81 all three
df = df[[17, 27, 73, 95, 18, 88, 55, 30, 41, 23, 64, 25, 29, 38, 57, 67, 77, 85, 68, 58, 24, 28, 46, 48, 84, 33, 62, 72, 34, 26, 15, 81]]
df = df.loc[[4, 9, 8, 10, 11]]

## R1.4 call function plot_pie_charts
plot_pie_chart(
    vals=data2plot_single.loc[17].dropna(),
    label=["Not Fulfilled", "Partially Fulfilled", "Fulfilled"],
    title="R14ConfigurationSelection",
)


# for idx_row, row_content in data2plot.iterrows():
# plot = row_content.plot.hist()
# plt.show()

print("ready")
