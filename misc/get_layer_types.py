from bs4 import BeautifulSoup
import requests
import re
import numpy as np
from matplotlib import pyplot as plt

url = "https://github.com/tensorflow/tensorflow/tree/v2.4.1/tensorflow/python/keras/applications"

page = requests.get(url)
soup = BeautifulSoup(page.content, "html.parser")

#find file names
file_names = soup.find_all("a", class_="js-navigation-open Link--primary")
ignore_files = ["BUILD", "__init__.py", "applications_load_weight_test.py", "applications_test.py"
                "imagenet_utils.py", "imagenet_utils_test.py"]
file_text = dict()

for file_name in file_names:
    #ignore certain files
    if file_name.text in ignore_files:
        continue

    #get code in file
    r = requests.get("https://raw.githubusercontent.com/tensorflow/tensorflow/v2.4.1/tensorflow/python/keras/applications/" + file_name.text)
    
    #remove single line comments
    text = re.sub("#.*", "", r.text)

    #remove multi line comments
    text = re.sub("(\"\"\")((?!\"\"\").)*(\"\"\")", "", text, flags=re.DOTALL)

    #add code to list for future parsing
    file_text[file_name.text] = text

#record which files a certain type of layer goes into
occurrence = dict()

#records how many total layer types there are in each of the files a type of layer is in
uniqueness = dict()

#record which layers are in specific files (basically occurence, inverted)
files = dict()

#certain layers may be already implemented, or just unnecessary for some reason
ignore_layers = ["Input"]

for file_name in file_text:
    #get layer names and clean the results
    r = re.findall("layers\.[^\(]*", file_text[file_name])
    layer_types = set([layer_type[7:] for layer_type in set(r)])

    #discard unnecessary layers
    for layer_name in ignore_layers:
        layer_types.discard(layer_name)

    #save file names
    for layer_name in layer_types:
        if layer_name not in occurrence.keys():
            occurrence[layer_name] = set()
            uniqueness[layer_name] = 0

        occurrence[layer_name].add(file_name)
        uniqueness[layer_name] += len(layer_types)

    #save layers to files
    files[file_name] = layer_types

#find final uniqueness of each layer type in file
for layer_name in uniqueness:
    uniqueness[layer_name] /= len(occurrence[layer_name])

layers = list()
x = list()
y = list()

#print data to output
for layer_name in occurrence:
    print(layer_name)
    print("\tOccurrence Count: " + str(len(occurrence[layer_name])))
    print("\tUniqueness Count: " + str(uniqueness[layer_name]))
    layers.append(layer_name)
    x.append(len(occurrence[layer_name]))
    y.append(uniqueness[layer_name])

#plot data
plt.title("Plot")
plt.xlabel("Occurrence Count")
plt.ylabel("Uniqueness")

plt.scatter(x, y)

for i in range(0, len(layers)):
    label = layers[i]
    plt.annotate(label, (x[i], y[i]), textcoords="offset points",
                 xytext=(0, 10), ha="center", size=10)

plt.show()