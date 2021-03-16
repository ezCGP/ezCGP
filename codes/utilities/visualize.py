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

    def add_to_csv(self, individual):
        self.csv_rows = []
        self.individual_num += 1
        prev_output = ""

        for block_num, block in enumerate(individual.blocks):
            curr_block = block

            shift = self.shifts[block_num]
            color = self.colors[block_num]
            for active_node in curr_block.active_nodes:
                fn = block.genome[active_node]

                if active_node < 0:  # Input
                    layer_info = "nickname= {}".format(
                        block.block_nickname)
                    out = "{}{}{},{},\"{}\",{},\"{}\",{}".format(
                        self.individual_num, shift, active_node, fn, layer_info, color, prev_output, self.arrow_color)
                elif type(fn) == np.int64:
                    out = "{}{}{},Output,,{},\"{}\",{}".format(self.individual_num, shift, active_node, color, str(
                        self.individual_num) + shift + str(fn), self.arrow_color)
                    prev_output = str(self.individual_num) + \
                        shift + str(active_node)
                else:
                    out = "{}{}{},{},,{},\"{}\",{}".format(self.individual_num, shift, active_node, fn['ftn'].__name__, color, ','.join(
                        map(lambda x: str(self.individual_num) + shift + str(x), fn['inputs'])), self.arrow_color)

                self.csv_rows.append(out + "")

        categorical_acc, precision, recall = individual.fitness.values
        self.csv_rows.append("END,\"Fitness: ({},{},{})\",,{},\"{}\",".format(
            -categorical_acc, -precision, -recall, '#ffe6cc', prev_output))

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
