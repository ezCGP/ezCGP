'''
root/code/utilities/selections.py

a few things grabbing from DEAP python module for evolution
https://deap.readthedocs.io/en/master/api/tools.html
'''

### packages
import deap.tools
import inspect

### sys relative to root dir
import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(dirname(realpath(__file__)))))

### absolute imports wrt root
from codes.utilities.custom_logging import ezLogging


def selNSGA2(individuals, k, nd='standard'):
    '''
    Non-Dominated Sorting   (NSGA-II)

    Apply NSGA-II selection operator on the *individuals*. Usually, the
    size of *individuals* will be larger than *k* because any individual
    present in *individuals* will appear in the returned list at most once.
    Having the size of *individuals* equals to *k* will have no effect other
    than sorting the population according to their front rank. The
    list returned contains references to the input *individuals*. For more
    details on the NSGA-II operator see [Deb2002]_.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param nd: Specify the non-dominated algorithm to use: 'standard' or 'log'.
    :returns: A list of selected individuals.

    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.

    From tests, looks like selNSGA2 maximises when Fitness.weights are positive
    '''
    return deap.tools.selNSGA2(individuals, k, nd)


def sortNondominated(individuals, k, first_front_only=False):
    '''
    Sort the first *k* *individuals* into different nondomination levels
    using the "Fast Nondominated Sorting Approach" proposed by Deb et al.,
    see [Deb2002]_. This algorithm has a time complexity of :math:`O(MN^2)`,
    where :math:`M` is the number of objectives and :math:`N` the number of
    individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param first_front_only: If :obj:`True` sort only the first front and
                             exit.
    :returns: A list of Pareto fronts (lists), the first list includes
              nondominated individuals.
    .. [Deb2002] Deb, Pratab, Agarwal, and Meyarivan, "A fast elitist
       non-dominated sorting genetic algorithm for multi-objective
       optimization: NSGA-II", 2002.
    '''
    return deap.tools.sortNondominated(individuals, k, first_front_only)


def selTournamentDCD(individuals, k):
    '''
    Tournament selection based on dominance (D) between two individuals, if
    the two individuals do not interdominate the selection is made
    based on crowding distance (CD). The *individuals* sequence length has to
    be a multiple of 4. Starting from the beginning of the selected
    individuals, two consecutive individuals will be different (assuming all
    individuals in the input list are unique). Each individual from the input
    list won't be selected more than twice.

    This selection requires the individuals to have a :attr:`crowding_dist`
    attribute, which can be set by the :func:`assignCrowdingDist` function.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :returns: A list of selected individuals.
    '''
    return deap.tools.selTournamentDCD(individuals, k)


def sortLogNondominated(individuals, k, first_front_only=False):
    '''
    Generalized Reduced runtime ND sort

    Sort *individuals* in pareto non-dominated fronts using the Generalized
    Reduced Run-Time Complexity Non-Dominated Sorting Algorithm presented by
    Fortin et al. (2013).

    :param individuals: A list of individuals to select from.
    :returns: A list of Pareto fronts (lists), with the first list being the
              true Pareto front.
    '''
    return deap.tools.sortLogNondominated(individuals, k, first_front_only)


def call_selection(method_name, **kwargs):
    '''
    so even if we don't provide the selection method above, as long as it is in the deap.tools library, we can call it here
    '''
    if not method_name in deap.tools.__dict__:
        ezLogging.critical("couldn't find selection method %s in deap.tools" % method_name)
        import pdb; pdb.set_trace()

    method = deap.tools.__dict__[method_name]
    if not inspect.isfunction(method):
        ezLogging.critical("given 'method_name' %s in deap.tools is not actually a function/method" % method_name)
        import pdb; pdb.set_trace()

    return method(**kwargs)