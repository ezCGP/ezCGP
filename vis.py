"""
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
"""

header = "## Hello World \
            \n# label: %step% \
            \n# style: shape=rectangle;rounded=1;fillColor=%fill%;strokeColor=#000000 \
            \n# namespace: csvimport-  \
            \n# connect: {\"from\":\"refs\", \"to\":\"id\", \"invert\":true, \"style\":\"curved=0;endArrow=blockThin;endFill=1;\"} \
            \n# width: auto \
            \n# height: auto \
            \n# padding: 10 \
            \n# ignore: id,fill,refs \
            \n# nodespacing: 10 \
            \n# levelspacing: 30 \
            \n# edgespacing: 40 \
            \n# layout: horizontalflow \
            \n## CSV starts under this line \
            \nid,step,fill, refs \
            \n "

class Visualizer():
    def __init__(self):
        self.shifts = ['a', 'b', 'c']
        self.colors = ['#dae8fc', '#f8cecc', '#ffe6cc']


    def create_csv(self, individual, output_path='vis.csv'):
        csv_rows = header.split("\n")

        prev_output = ""
        for block_num in range(individual.num_blocks):
            curr_block = individual.skeleton[block_num+1]["block_object"]
            shift = self.shifts[block_num]
            color = self.colors[block_num]
            for active_node in curr_block.active_nodes:
                fn = curr_block[active_node]
                if active_node < 0:  #Input
                    out = "{}{},{},{},\"{}\"".format(shift, active_node, fn, color, prev_output)
                elif active_node >= curr_block.genome_main_count:
                    out = "{}{},Output,{},\"{}\"".format(shift, active_node, color, shift + str(fn))
                    prev_output = shift + str(active_node)
                else:
                    out = "{}{},{},{},\"{}\"".format(shift, active_node, fn['ftn'].__name__, color, ','.join(map(lambda x: shift + str(x), fn['inputs'])))
                csv_rows.append(out + "")

        csv_rows.append('END,"Fitness: {}",{},\"{}\"'.format(individual.fitness.values, '#ffe6cc', prev_output))

        with open(output_path, 'w') as csv:
            for row in csv_rows:
                csv.write(row + '\n')

