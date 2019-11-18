#This is going to be the file that converts to csv and uploads to draw.io
'''
This class takes in an individual and writes it to a .csv for draw.io to process

Use: Instantiate object, then pass an individual to create_csv
'''
class Visualizer():
    def __init__(self):

    def create_csv(self, individual, output_path='vis.csv'):
        csv_rows = []
        for i in range(1, individual.num_blocks+1):
            row = []
            curr_block = individual.skeleton[i]["block_object"]
            row.append(self.get_args(curr_block))
            for node in curr_block.active_nodes:
                #TODO seperate input/middle/output nodes
                block_fn = curr_block[node]
                row.append('{}'.format(fn))
            csv_rows.append(row)

    def write_file(self, csv_rows, output_path):
        csv = open(output_path, 'w')
        #TODO label the columns and add colors and stuff idk
        for row in csv_rows:
            for item in row:
                csv.write(item+',')
            csv.write('\n')
        csv.close()


    def get_args(self, block):
        #TODO this should definitely be in the block itself
        #TODO actually get the arguments in a nice way
        return block.arg_values
