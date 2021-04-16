"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
"""

import pickle
import string
import csv
import numpy as np

header = '## Hello World \
            \n# label: %step%<br><i style="color:gray;">%text%</i> \
            \n# style: html=1;shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=%fill% \
            \n# namespace: csvimport- \
            \n# connect: {\"from\":\"refs\", \"to\":\"id\", \"invert\":true, \"style\":\"curved=0;endArrow=blockThin;endFill=1;fillColor=#1500ff;strokeColor=#1500ff;\"} \
            \n# width: auto \
            \n# height: auto \
            \n# padding: 10 \
            \n# ignore: id,fill,refs \
            \n# nodespacing: 10 \
            \n# levelspacing: 30 \
            \n# edgespacing: 40 \
            \n# layout: horizontalflow \
            \n## CSV starts under this line \
            \nid,step,text,fill,refs,arrow_color \
            \n '


class Visualizer():
    def __init__(self, output_path='vis.csv'):
        # Limit on number of block is letters of alphabet
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9
        self.header = header
        self.arrow_color = "#1500ff"
        self.csv_rows = self.header.split("\n")
        self.individual_num = 0
        self.append_csv(True)

    def add_to_csv(self, individual, print_entire_genome=False):
        self.csv_rows = []
        self.individual_num += 1
        prev_output = ''
        for block_num, block in enumerate(individual.blocks):
            shift = self.shifts[block_num]
            color = self.colors[block_num]

            first_node = block.active_nodes[0]
            for current_node in range(first_node, len(block.genome) + first_node):
                if current_node not in block.active_nodes:
                    if not print_entire_genome:
                        continue
                    color = '#808080'    # should this color be hard-coded?
                else:
                    color = self.colors[block_num]
                fn = block.genome[current_node]

                if current_node < 0:  # Input
                    layer_info = "nickname= {}".format(
                        block.block_nickname)
                    out = "{}{}{},{},\"{}\",{},\"{}\",{}".format(
                        self.individual_num, shift, current_node, fn, layer_info, color, prev_output, self.arrow_color)
                elif type(fn) == np.int64:
                    out = "{}{}{},Output,,{},\"{}\",{}".format(self.individual_num, shift, current_node, color, str(
                        self.individual_num) + shift + str(fn), self.arrow_color)
                    prev_output = str(self.individual_num) + \
                        shift + str(current_node)
                else:
                    arg_str = []
                    for arg in fn['args']:
                        value = block.args[arg].value
                        if hasattr(value, "__name__"):
                            value = value.__name__
                        arg_str.append(str(value))

                    layer_info = "args= {}".format(", ".join(arg_str))
                    out = "{}{}{},{},\"{}\",{},\"{}\",{}".format(self.individual_num, shift, current_node, fn['ftn'].__name__, layer_info, color, ','.join(
                        map(lambda x: str(self.individual_num) + shift + str(x), fn['inputs'])), self.arrow_color)

                self.csv_rows.append(out + "")

        accuracy, precision, recall = individual.fitness.values
        self.csv_rows.append(f'END,\"Fitness: ({-accuracy},{-precision},{-recall})\",,#ffe6cc,\"{prev_output}\",')
        self.append_csv()

    def append_csv(self, new=False):
        import os
        if new:
            ext = 0
            while os.path.isfile(self.output_path):
                ext += 1
            self.output_path = self.output_path + "_" + str(ext)

        with open(self.output_path + ".csv", 'a+') as csv:
            for row in self.csv_rows:
                csv.write(row + '\n')


with open('gen_0007_indiv_2e5edc094.pkl', 'rb') as indiv_pkl:
    indiv = pickle.load(indiv_pkl)
    viz = Visualizer()
    viz.add_to_csv(indiv)
