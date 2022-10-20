import geopandas as gpd
import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import itertools
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use("seaborn-white")


def count_countries(dict_country):
    countries = dict_country.values()

    countries_list = list(countries)

    flat_countries_list = []
    for entries in countries_list:
        if isinstance(entries, list):
            for entry in entries:
                flat_countries_list.append(entry)
        else:
            flat_countries_list.append(entries)
    authors_countries = {c: flat_countries_list.count(c) for c in flat_countries_list}

    return authors_countries


def plot_pie_chart(vals, label, title):
    fig, ax = plt.subplots()
    ax.pie(
        vals,
        labels=label,
        autopct="%1.1f%%",
        startangle=90,
        colors=plt.colormaps["Blues"](
            [int(255 - 205 * 1 / x) for x in np.arange(1, len(vals) + 1)]
        ),
    )
    #ax.set_title(title)
    plt.savefig(title + '.pdf', bbox_inches='tight', pad_inches=0)
    plt.close()


def count_combinations(rows):
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


# main

# create dictionary from json
with open("authors_origin.json", "r") as author_origin:
    dict_country = json.load(author_origin)

with open("included_papers.json") as include_paper:
    included_papers = json.load(include_paper)
    included_papers = [str(int) for int in included_papers]

remove_paper = []

for key in dict_country:
    if key in included_papers:
        pass
    else:
        remove_paper.append(key)

for paper in remove_paper:
     del dict_country[paper]


dict_country = {int(k): v for k, v in dict_country.items()}



# call function count_countries
authors_countries = count_countries(dict_country)

# create world plot colored by amount of papers/authors from that country
## load df of the world from geopandas
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

## add column to the world dictionary to paste the numbers of authors
countries["authors_count"] = None

for key, value in authors_countries.items():
    assert isinstance(value, object)
    row = countries.index[countries["name"] == key]
    countries.loc[row, "authors_count"] = value

count_numbers = []
for idx, row in countries.iterrows():
    count_numbers.append(row["authors_count"])

## exclude Antarctica from map
countries = countries[countries.name != "Antarctica"]

## create List with author count and None for labeling
count_label = countries["authors_count"].to_list()

## style the map and plot it
ax = countries.plot(
    column="authors_count",
    missing_kwds={"color": "lightgrey", "label": "Non"},
    cmap="Wistia",
    edgecolor="darkgrey",
    legend=True,
    figsize=(15,10),
)
#ax.set_axis_off(),
plt.savefig("World.pdf")
plt.close()

# load and clean analysis data
filepath = "./2022-10-18 Full_Text_RM_Survey_cleaned.csv"

data = pd.read_csv(filepath, sep=";", header=None)

for idx_row, row_content in data.iterrows():
    for idx_col, content in row_content.iteritems():
        if isinstance(content, str):
            if idx_row == 0:
                if content.startswith("# "):
                    data.loc[idx_row, idx_col] = int(content[2:4])
                elif content.startswith("#"):
                    data.loc[idx_row, idx_col] = int(content[1:3])
            else:
                try:
                    data.loc[idx_row, idx_col] = int(content[:1])
                except ValueError:
                    pass


data2plot = data.drop(index=[0, 22, 23, 24]).dropna()

# count and sort data for pie plots
data2plot_single = data2plot.apply(pd.Series.value_counts, axis=1)

# data to latex table
data2tex = data.drop(index=[22,23,23]).dropna()
data2tex = pd.DataFrame(data2tex.T)

with open('Result_Table.tex', 'w') as tab:
    tab.write(data2tex.to_latex(index=False))
#columns=['R1', 'R1.1', 'R1.2', 'R1.2.1', 'R1.2.1.1', 'R1.2.1.2', 'R1.2.2', 'R1.2.2.1', 'R1.2.2.2', 'R1.2.2.3', 'R1.2.2.4', 'R1.2.3', 'R1.3', 'R1.3.1', 'R1.3.2', 'R1.3.3', 'R1.4', 'R2', 'R3', 'R4', 'R5']


# R1

## R1.2.1 call function plot_pie_charts
plot_pie_chart(
    vals=data2plot_single.loc[5].dropna(),
    label=["Not Considered", "Partially Considered", "Fully Considered"],
    title="R121HardwareConsideration",
)

plot_pie_chart(
    vals=data2plot_single.loc[6].dropna(),
    label=["Not Considered", "Partially Considered", "Fully Considered"],
    title="R121SoftwareConsideration",
)

## R1.2.1 nested pie chart -> needs to be cleaned up and generalized
subgroup_names=['3%', '31%', '','25%', '17%', '','21%']
#['HW and SW are fully considered', 'HW or SW is fully considered','','HW and SW are partially considered', 'HW or SW is partially considered','','Non is considered',]
subgroup_size=[1,9,0,7,5,0,6]
group_names=['','SW 17%','HW 14%','','SW 10%','HW 7%','']
group_size=[1,5,4,7,3,2,6]

# Create colors
a, b, c, d, e =[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens, plt.cm.Oranges, plt.cm.Purples]

# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, labeldistance=0.78, colors=
['white', b(0.5), b(0.4),'white', d(0.5), d(0.4),'white'])
plt.setp(mypie, width=0.5, edgecolor='white')

# Second Ring (Inside)
mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.5,
labels=subgroup_names, labeldistance=0.68, colors=[a(0.6), b(0.6), b(0.6),c(0.6), d(0.6),d(0.6), e(0.6)])
plt.setp(mypie2, width=0.5, edgecolor='white')
plt.margins(-0.4,5, tight=False)

plt.legend(loc=(0.9, 0.1))
handles, labels = ax.get_legend_handles_labels()

subgroup_names_legs=['HW and SW are fully considered', 'HW or SW is fully considered','HW and SW are partially considered', 'HW or SW is partially considered','Non is considered']

ax.legend(handles[4:], subgroup_names_legs, loc=(0.6, 0))
plt.savefig("R121nestedPie.pdf")

## R1.2.3 call function plot_pie_charts
plot_pie_chart(
    vals=data2plot_single.loc[12].dropna(),
    label=["Not Considered", "Partially Considered", "Fully Considered"],
    title="R123IntelligentSolutionSpace",
)


# for idx_row, row_content in data2plot.iterrows():
# plot = row_content.plot.hist()
# plt.show()

print("ready")
