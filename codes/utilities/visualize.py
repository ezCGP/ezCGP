"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
"""
import argparse
import pickle
import string
import csv
import os
import sys
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


class Visualizer:
    def __init__(self, output_path='vis'):
        # Limit on number of block is letters of alphabet
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9
        self.header = header
        self.arrow_color = '#1500ff'
        self.csv_rows = self.header.split('\n')
        self.individual_num = 0
        self.append_csv(True)

    def add_to_csv(self, individual, print_entire_genome):
        self.csv_rows = []
        self.individual_num += 1
        prev_output = ''
        for block_num, block in enumerate(individual.blocks):
            shift = self.shifts[block_num]
            color = self.colors[block_num]

            first_node = block.active_nodes[0]
            for index in range(first_node, len(block.genome) + first_node):
                if index not in block.active_nodes:
                    if not print_entire_genome:
                        continue
                    color = '#f8cecc'    # should this color be hard-coded?
                else:
                    color = self.colors[block_num]
                fn = block.genome[index]

                if index < 0:  # Input
                    layer_info = f'nickname= {block.block_nickname}'
                    out = f'{self.individual_num}{shift}{index},{fn},\"{layer_info}\",{color},\"{prev_output}\",{self.arrow_color}'
                elif type(fn) == np.int64:
                    output = f'{self.individual_num}{shift}{fn}'
                    out = f'{self.individual_num}{shift}{index},Output,,{color},\"{output}\",{self.arrow_color}'
                    prev_output = f'{self.individual_num}{shift}{index}'
                else:
                    inputs = ','.join([f'{self.individual_num}{shift}{x}' for x in fn['inputs']])
                    out = f'{self.individual_num}{shift}{index},{fn["ftn"].__name__},,{color},\"{inputs}\",{self.arrow_color}'
                self.csv_rows.append(out + "")

        accuracy, precision, recall = individual.fitness.values
        self.csv_rows.append(f'END,\"Fitness: ({-accuracy},{-precision},{-recall})\",,#ffe6cc,\"{prev_output}\",')
        self.append_csv()

    def append_csv(self, new=False):
        if new:
            ext = 0
            while os.path.isfile(self.output_path):
                ext += 1
            self.output_path = f'{self.output_path}_{ext}'

        with open(f'{self.output_path}.csv', 'a+') as f:
            for row in self.csv_rows:
                f.write(row + '\n')


if __name__ == '__main__':
    sys.path.append('../../../../')
    parser = argparse.ArgumentParser(description="Navigate to output folder containing individual pickle files and "
                                                 "call this function using 'python "
                                                 "../../../../codes/utilities/visualize.py <individual.pkl>'")
    parser.add_argument('filepath', help='Name of the individual that you want to visualize.')
    parser.add_argument('-v', help='Show unused nodes', action='store_true')
    args = parser.parse_args()

    with open(args.filepath, 'rb') as f:
        individual = pickle.load(f)
        viz = Visualizer()
        viz.add_to_csv(individual, args.v == True)
