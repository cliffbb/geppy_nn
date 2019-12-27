"""This module 'compile' provides a method translating list of graphs consist of nodes and edges
into computational graph expressions that can be used in convolutional neural network (CNN)
frameworks such as PyTorch and TensorFlow.
"""
def create_expr(ops):
    return 'self.{}(x)'.format(ops[2])

def add(ops, graph_dict):
    return 'self.{}(add({}))'.format(ops[2], ', '.join(graph_dict[j] for j in ops[1]))

def add_out(ops, graph_dict):
    return 'add({})'.format(', '.join(graph_dict[j] for j in ops[1]))

def without_add(ops, graph_dict):
    return 'self.{}({})'.format(ops[2], graph_dict[ops[1][0]])

def concat(ops, graph_dict):
    return 'concat({})'.format(', '.join(graph_dict[j] for j in ops[1]))

def no_op(ops, graph_dict):
    return '{}'.format(graph_dict[ops[1][0]])

def compile_graph(graph_list):
    graph = {}
    for operation in graph_list:
        if operation[1][0] == 'input':
            graph[operation[0]] = create_expr(operation)
        elif operation[0] != 'output' and len(operation[1]) == 1:
            graph[operation[0]] = without_add(operation, graph)
        elif operation[0] != 'output' and len(operation[1]) > 1:
            graph[operation[0]] = add(operation, graph)
        elif operation[0] == 'output' and len(operation[1]) > 1:
            graph[operation[0]] = add_out(operation, graph)
        elif operation[0] == 'output' and len(operation[1]) == 1:
            graph[operation[0]] = no_op(operation, graph)
        else:
            raise NotImplementedError('Unimplemented operation: ', operation)
    return graph['output']

# exported function
__all__ = ['compile_graph']
