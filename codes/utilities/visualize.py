"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
"""
import argparse
import glob
import os
import pickle
import random
import string
import sys

import numpy as np


HEADER = '## Hello World \
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
INPUT_ROW = '{}{}{},{},\"nickname = {}\",{},\"{}\",{}'
NORMAL_ROW = '{}{}{},{},\"node #{}\",{},\"{}\",{}'
OUTPUT_ROW = '{0}{1}{2},Output,,{3},\"{0}{1}{4}\",{5}'
END_ROW = 'END,\"Fitness: ({},{},{})\",,#ffe6cc,\"{}\",'
INACTIVE_NODE_COLOR = '#f8cecc'


class Visualizer:
    def __init__(self, output_path='vis'):
        # Limit on number of block is letters of alphabet
        self.output_path = output_path
        self.shifts = list(string.ascii_lowercase)
        self.colors = ['#dae8fc', '#f8cecc', "#d5e8d4"] * 9
        self.header = HEADER
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
            indices = block.active_nodes
            if print_entire_genome:
                indices = range(indices[0], len(block.genome) + indices[0])
            for index in indices:
                color = self.colors[block_num] if index in block.active_nodes else INACTIVE_NODE_COLOR
                fn = block.genome[index]
                if index < 0:  # Input
                    out = INPUT_ROW.format(self.individual_num, shift, index, fn, block.block_nickname, color, prev_output, self.arrow_color)
                elif type(fn) == np.int64:
                    out = OUTPUT_ROW.format(self.individual_num, shift, index, color, fn, self.arrow_color)
                    prev_output = f'{self.individual_num}{shift}{index}'
                else:
                    inputs = ','.join([f'{self.individual_num}{shift}{x}' for x in fn['inputs']])
                    out = NORMAL_ROW.format(self.individual_num, shift, index, fn["ftn"].__name__, index, color, inputs, self.arrow_color)
                self.csv_rows.append(out + "")
        accuracy, precision, recall = individual.fitness.values
        self.csv_rows.append(END_ROW.format(-accuracy, -precision, -recall, prev_output))
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
    parser = argparse.ArgumentParser(description="Navigate to univ0000 folder containing individual pickle files and "
                                                 "call this function using 'python "
                                                 "../../../../codes/utilities/visualize.py'")
    parser.add_argument('individual', help='Name of the individual that you want to visualize. If not provided, '
                                           'all individuals are visualized.', nargs='?', default='*.pkl')
    parser.add_argument('-i', help='Show inactive nodes.', action='store_true')
    args = parser.parse_args()

    viz = Visualizer()
    # Select 20 random individuals (only applicable if 'individual' argument not passed).
    individuals = glob.glob(args.individual)
    random.shuffle(individuals)
    for individual in individuals[:20]:
        with open(individual, 'rb') as f:
            individual = pickle.load(f)
            viz.add_to_csv(individual, args.i)
    print("CSV successfully created!")
