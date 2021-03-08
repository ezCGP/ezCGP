'''
take in a list of directories from which to grab the npz files for ploting
'''
### packages
import glob
import matplotlib.pyplot as plt


### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))

### absolute imports wrt root
from post_process import plot_things


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--minimization",
                        action = "store_const",
                        const = True,
                        default = False,
                        help = "plot a maximization pareto front instead of minimization")
    parser.add_argument("-d",'--dirs',
                        nargs='+', # 1 or more
                        help='List directories for each universe run',
                        required=True)
    args = parser.parse_args()

    fig, axis = plt.subplots(1, 1, figsize=(10,15))

    generation = 0
    for dir_path in args.dirs:
        npzs = glob.glob(os.path.join(dir_path, "gen*_fitness.npz"))
        for npz in npzs:
            plot_things.plot_pareto_front_from_fitness_npz(axis,
                                                           npz,
                                                           minimization=args.minimization,
                                                           color=plot_things.matplotlib_colors[generation],
                                                           label="Gen %2d" % generation,
                                                          )
            generation+=1

    plt.show()