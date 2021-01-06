"""
This module 'algorithm' uses DEAP platform for the implementation of GEP evolution process. The
implementation is based on 'deap.base.Toolbox' object.

Note:: the code is adopted from Shuhua Gao @
https://geppy.readthedocs.io/en/latest/_modules/geppy/algorithms/basic.html#gep_simple
"""
import deap
import random
import time
import warnings


def _validate_toolbox(tb):
    """
    Validate the operators in the deap toolbox 'tb' according to our naming conventions.
    """
    assert hasattr(tb, 'select'), "The toolbox must have a selection operator 'select'."

    # check if all operators in 'tb.pbs' are registered
    for op in tb.pbs:
        assert op.startswith('mut') or op.startswith('cx'), "Mutation operator must begin with 'mut' and " \
                                                            "crossover with 'cx' except selection."
        assert hasattr(tb, op), "The operator '{}' is not registered in the toolbox, but a probability " \
                                "value is specified.".format(op)

    # check if all the mutation and crossover operators have their probabilities assigned in 'tb.pbs'
    for op in [attr for attr in dir(tb) if attr.startswith('mut') or attr.startswith('cx')]:
        if op not in tb.pbs:
            warnings.warn('Operator {0} has no probability value assigned. By default, the probability is '
                          'zero and the operator {0} will not be applied.'.format(op), category=UserWarning)


def _apply_mutation(pop, op, pb):
    """
    Apply a mutation genetic operation 'op' to each individual in population 'pop' with probability 'pb'.
    """
    for i in range(len(pop)):
        if random.random() < pb:
            pop[i], = op(pop[i])
            del pop[i].fitness.values
    return pop


def _apply_crossover(pop, op, pb):
    """
    Apply a crossover genetic operation 'op' to each individual in population 'pop' with probability 'pb'.
    """
    for i in range(1, len(pop), 2):
        if random.random() < pb:
            pop[i - 1], pop[i] = op(pop[i - 1], pop[i])
            del pop[i - 1].fitness.values
            del pop[i].fitness.values
    return pop


def gep_EA(pop, toolbox, gen_days, n_elites=1, stats=None, hof=None, history=None, verbose=__debug__):
    """
    This algorithm performs the simplest and standard gene expression programming.
    The flowchart of this algorithm can be found @
    <https://www.gepsoft.com/gxpt4kb/Chapter06/Section1/SS1.htm>.
    Refer to Chapter 3 of 'Gene Expression Programming: Mathematical Modeling by an Artificial Intelligence'
    by Candida Ferreira to learn more about this basic algorithm.

    :param pop: a list of individuals
    :param toolbox: class '~geppy.tools.toolbox.Toolbox', a container of operators.
    :param gen_days: max number of generation days to evolve
    :param n_elites: number of elites to be cloned to next generation
    :param stats: a class '~deap.tools.Statistics' object that is updated inplace, optional.
    :param hof: a class '~deap.tools.HallOfFame' object that will contain the best individuals, optional.
    :param history: a class '~deap.tools.History' object that will contain the history of the individuals.
    :param verbose: whether or not to print the statistics.
    :returns: The final population
    :returns: A :class:'~deap.tools.Logbook' recording the statistics of the evolution process
    """
    _validate_toolbox(toolbox)
    logbook = deap.tools.Logbook()
    logbook.header = ['n_gen', 'n_eval'] + (stats.fields if stats else [])

    hours = 0
    n_gen = 1
    start = time.time()
    while hours < gen_days*24:
        # evaluate only the invalid individuals i.e. no need to reevaluate the unchanged individuals
        invalid_indvs = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_indvs)
        for ind, fit in zip(invalid_indvs, fitnesses):
            ind.fitness.values = fit

        # record statistics and log
        if hof is not None:
            hof.update(pop)
        record = stats.compile(pop) if stats else {}
        logbook.record(n_gen=n_gen, n_eval=len(invalid_indvs), **record)
        if verbose:
            print(logbook.stream)

        # log individual history
        if history is not None:
            pass

        # selection with elitism
        elites = deap.tools.selBest(pop, k=n_elites)
        offspring = toolbox.select(pop, len(pop) - n_elites)

        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_mutation(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        pop = elites + offspring

        n_gen += 1
        hours = divmod((time.time() - start), 3600)[0]

    return pop, logbook


__all__ = ['gep_EA']