"""This module 'compile' provides a method translating list of graphs consist of nodes and edges
into computational graph expressions that can be used in convolutional neural network (CNN)
frameworks such as PyTorch and TensorFlow.
"""
def unit(ops):
    return 'self.{}(x)'.format(ops[2][0])

def in_add(ops, graph_dict):
    return 'self.{}(add(x, {}))'.format(ops[2][0], ', '.join(graph_dict[j] for j in ops[1][1:]))

def add(ops, graph_dict):
    return 'self.{}(add({}))'.format(ops[2][0], ', '.join(graph_dict[j] for j in ops[1]))

def out_add(ops, graph_dict):
    return 'add({})'.format(', '.join(graph_dict[j] for j in ops[1]))

def dual(ops, graph_dict):
    return 'self.{}({})'.format(ops[2][0], graph_dict[ops[1][0]])

def concat(ops, graph_dict):
    return 'concat({})'.format(', '.join(graph_dict[j] for j in ops[1]))

def no_op(ops, graph_dict):
    return '{}'.format(graph_dict[ops[1][0]])

def compile_graph(graph_list):
    graph = {}
    for operation in graph_list:
        if operation[1][0] == 'input'and len(operation[1]) == 1:
            graph[operation[0][0]] = unit(operation)
        elif operation[1][0] == 'input' and len(operation[1]) > 1:
            graph[operation[0][0]] = in_add(operation, graph)
        elif operation[0][0] != 'output' and len(operation[1]) == 1:
            graph[operation[0][0]] = dual(operation, graph)
        elif operation[0][0] != 'output' and len(operation[1]) > 1:
            graph[operation[0][0]] = add(operation, graph)
        elif operation[0][0] == 'output' and len(operation[1]) > 1:
            graph[operation[0][0]] = out_add(operation, graph)
        elif operation[0][0] == 'output' and len(operation[1]) == 1:
            graph[operation[0][0]] = no_op(operation, graph)
        else:
            raise NotImplementedError('Unimplemented operation: ', operation)
    #print(graph)
    return graph['output']

# exported function
__all__ = ['compile_graph']
