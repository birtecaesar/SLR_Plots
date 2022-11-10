import geopandas as gpd
import matplotlib.pyplot as plt
import json

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
    missing_kwds={"color": "lightgrey", "label": "None"},
    cmap="Wistia",
    edgecolor="darkgrey",
    legend=True,
    figsize=(15, 10),
)
# ax.set_axis_off(),
plt.savefig("World.pdf")
plt.close()