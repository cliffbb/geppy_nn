"""This module 'cell_graph' provides the basic operators to build neuronal cells graph. It contains
function useful to create neuronal-cell directed acyclic graph (DAG). The module is based on
*AGraph* module of pygraphviz package. The module provides five basic program-symbol (PAR, SEQ, CPO, CPI,
END) functions in cellular encoding for creating neuronal-cell graph.
    **Ref: Gruau, F. (1994) Neural Network Synthesis Using Cellular Encoding and the
           Genetic Algorithm Ph.D. Thesis, l’Ecole Normale Superieure de Lyon.
    **Ref: Whitley, D., Gruau, F., and Pyeatt, L. (1995) Cellular Encoding Applied to Neurocontrol in
           the Proceedings of the Sixth International Conference (ICGA95)
"""
import os
import networkx as nx
from pygraphviz import AGraph
from gepcore.utils.compile import compile_graph
from gepcore.entity import KExpressionGraph, Chromosome

def par(graph, cell1, cell2):
    """Get cellular encoding program-symbol *PAR* function. Add new node *cell2* to a DAG *graph*. The new
    node *cell2* shares same input and output with node *cell1* which is a node already in the DAG *graph*.
    The new node *cell2* is placed parallel to the node *cell1*.
    :param graph: obj, an AGraph object
    :param cell1: str, name of a node 1
    :param cell2: str, name of node 2
    """
    cell2_in = list(graph.predecessors(cell1))
    cell2_out = list(graph.successors(cell1))
    for cell in cell2_in:
        graph.add_edge(cell, cell2)
    for cell in cell2_out:
        graph.add_edge(cell2, cell)


def seq(graph, cell1, cell2):
    """Get cellular encoding program-symbol *SEQ* function. Add new node *cell2* to a DAG *graph* and
    connect it to node *cell1* which is already in the DAG *graph*. The new node *cell2* takes the output
    of node *cell1* as input, and connects its output to the former successor of node *cell1*. The new node
    *cell2* is placed sequential to the node *cell1* which is already in the DAG *graph*.
    :param graph: obj, an AGraph object
    :param cell1: str, name of a node 1
    :param cell2: str, name of node 2
    """
    cell2_out = list(graph.successors(cell1))
    for cell in cell2_out:
        graph.add_edge(cell2, cell)
        graph.remove_edge(cell1, cell)
    graph.add_edge(cell1, cell2)


def cpi(graph, cell1, cell2):
    """Get cellular encoding program-symbol *SEQ* function. Add new node *cell2* to a DAG *graph* and
    connect it to node *cell1* which is already in the DAG *graph*. The new node *cell2* takes the output
    of node *cell1* as input, and connects its output to the former successor of node *cell1*. The new node
    *cell2* is placed sequential to the node *cell1* which is already in the DAG *graph*.
    :param graph: obj, an AGraph object
    :param cell1: str, name of a node 1
    :param cell2: str, name of node 2
    """
    cell2_in = list(graph.predecessors(cell1))
    cell2_out = list(graph.successors(cell1))
    for cell in cell2_in:
        graph.add_edge(cell, cell2)
    for cell in cell2_out:
        graph.add_edge(cell2, cell)
        graph.remove_edge(cell1, cell)
    graph.add_edge(cell1, cell2)


def cpo(graph, cell1, cell2):
    """Get cellular encoding program-symbol *CPO* function. Add new node *cell2* to a DAG *graph* and
    connect it to node *cell1* which is already in the DAG *graph*. The new node *cell2* shares same output
    with node *cell1* but also takes the output of node *cell1* as input.
    :param graph: obj, an AGraph object
    :param cell1: str, name of a node 1
    :param cell2: str, name of node 2
    """
    cell2_out = list(graph.successors(cell1))
    for cell in cell2_out:
        graph.add_edge(cell2, cell)
    graph.add_edge(cell1, cell2)


def end():
    """Get dummy implementation of cellular encoding end program-symbol *END*.
        Note: This actual operation is implicitly implemented in the genotype-phenotype mapping algorithm.
    """
    pass


def _generate_cell_graph(kexpr_graph):
    """Get a directed acyclic graph of a kexpression graph.
    :param kexpr_graph: list, list of edges and dict of nodes with labels
    :return: obj, an AGraph object
    """
    agraph = AGraph(directed=True)
    agraph.add_node('input', label='input')
    agraph.add_node('output', label='output')
    node_label = kexpr_graph[0]
    edges = kexpr_graph[1]

    init_edges = [('input', edges[0]), (edges[0], 'output')]
    agraph.add_edges_from(init_edges)

    for i, n in enumerate(node_label):
        agraph.add_node(n, label=node_label[n] + '_' + str(i))

    for edge, func in enumerate(edges):
        if edge > 0:
            func = str(func).replace('(', '(agraph,')
            eval(func)
    return agraph


def generate_cell_graph(genome):
    """Get a directed acyclic graph (DAG) of a genome/chromosome
    :param genome: obj, genome/chromosome consists of genes
    :return: obj, an AGraph object
    """
    kexpr_graph = KExpressionGraph.from_genotype(genome)
    if len(kexpr_graph) == 1:
        kexpr_graph = kexpr_graph[0]
        return [_generate_cell_graph(kexpr_graph)]
    elif len(kexpr_graph) > 1:
        agraph = []
        for i in range(len(kexpr_graph)):
            agraph.append(_generate_cell_graph(kexpr_graph[i]))
        return agraph


def postorder_traverse(graph, root='input'):
    """Get the computational sequence of directed acyclic graph
    :param graph: obj, Agraph object
    :param root: str, root of the graph
    :return: tuple, list of computational order and operation labels
    """
    comp_order = []
    visited_nodes = set()
    nodes_label = []
    inputs = [graph.get_node(n).attr['label'] for n in graph.successors('input')]

    def dfs_traverse(node):
        visited_nodes.add(node)
        if node not in ['input', 'output']:
            nodes_label.append(graph.get_node(node).attr['label'])

        for succ in graph.successors(node):
            if succ not in visited_nodes:
                dfs_traverse(succ)

        if len(graph.predecessors(node)) != 0:
            labels = [graph.get_node(n).attr['label'] for n in graph.predecessors(node)]
            comp_order.append([[node], graph.predecessors(node), [graph.get_node(node).attr['label']],
                               labels])
    dfs_traverse(root)
    return inputs, nodes_label, comp_order


def generate_comp_graph(genome):
    """Get computational graphs expression  of genome/chromosome. Each gene in a chromosome is
    expressed as a computational graph that can be implemented in convolutional neural network
    frameworks such as PyTorch and TensorFlow.
    :param genome: obj, genome/chromosome consists of genes
    :return: dict, an ordered dictionary of list of AGraph objects and their computational graphs
    """
    if isinstance(genome, Chromosome):
        graph = generate_cell_graph(genome)
    elif isinstance(genome, AGraph):
        graph = [genome]
    else:
        graph = genome

    comp_graph_obj = []
    comp_graph_expr = []

    for g in graph:
        comp_graph_obj.append(g)
        inputs, labels, comp_order = postorder_traverse(g)
        comp_order.reverse()
        comp_graph_expr.append([inputs, labels, compile_graph(comp_order)])
    return comp_graph_obj, comp_graph_expr


def draw_graph(graph, dir_path):   # have to add more graph features
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    g = nx.compose_all([nx.nx_agraph.from_agraph(g) for g in graph])
    g = nx.nx_agraph.to_agraph(g)
    g.draw(dir_path+'/comp_graph.png', format='png', prog='dot')


def save_graph(graph, dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    for i, g in enumerate(graph):
        g.write(dir_path+'/gene_{}.dot'.format(i))


def nx_agraph(graph):
    g = nx.compose_all([nx.nx_agraph.from_agraph(g) for g in graph])
    return nx.nx_agraph.to_agraph(g)

# functions exported
__all__ = ['par', 'seq', 'end', 'cpi', 'cpo', 'generate_cell_graph', 'generate_comp_graph',
           'draw_graph', 'save_graph']
