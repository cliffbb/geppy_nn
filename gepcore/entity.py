"""This module 'ea_entity' encapsulates the necessary evolutionary algorithm (EA) entities data
structures used in GEP-NN, which include the gene, chromosome (genotype), and neuronal-cell
graph (phenotype). A genotype is made up of primitives which consist of cellular encoding
program-symbols and neuronal-cell symbols called genetic codes.

A chromosome composed of one or more genes, which are of GEP-like K-Expression fixed length
linear string
    **Ref: Ferreira, Cândida. Gene expression programming: mathematical modeling by an artificial
           intelligence. Vol. 21. Springer, 2006
A phenotype is a digraph of neuronal-cell symbols which are basic convolution operators of CNN
    **Ref: NASNet search space (Zoph, et. al., 2018).
The class *Chromosome* in this module extend the class *Chromosome* in the module *entity*
in geppy.
    **Ref: <https://geppy.readthedocs.io/en/latest/geppy.core.html#module-geppy.core.entity>
"""
from geppy.core import entity
import random

def gene_generator(pset, head_length):
    """Generate a fixed length linear string gene randomly from a primitive set *pset*"""
    functions = pset.ps_functions
    terminal = pset.ps_terminal
    cells = pset.cell_symbols
    cell_length = head_length * (pset.max_arity - 1) + 1
    gene_len = head_length + cell_length
    gene = [None] * gene_len

    for i in range(head_length):
        gene[i] = random.choice(functions + terminal)
    for i in range(head_length, gene_len):
        gene[i] = random.choice(cells)
    return gene


class Gene(list):
    """Class that represents a fixed length linear string gene made up of cellular encoding
    program-symbol functions and neuronal-cell symbols.
    """
    def __init__(self, pset, head_length):
        """Instantiate a gene made up of program-symbols and neuronal-cell symbols with the length =
        head_length + cell_length + tail_length, where tail_length = head_length * (max_arity - 1) + 1
        and cell_length = tail_length
        :param pset: obj, *PrimitiveSet* class object that contains gene primitive set
        :param head_length: int, length of functional program-symbols
        """
        self._max_arity = pset.max_arity
        self._head_length = head_length
        genome = gene_generator(pset, head_length)
        super(Gene, self).__init__(genome)

    @classmethod
    def from_gene(cls, genome, head_length):
        """Create a gene from the given *gene*.
        :param genome: list, a list of program-symbols and neuronal-cell symbols
        :param head_length: length of the head domain of the program-symbols
        :return: obj, a gene
        """
        gene = super().__new__(cls)
        super.__init__(genome)
        gene._head_length = head_length
        return gene

    @property
    def head_length(self):
        """Get the length of the head section of a gene"""
        return self._head_length

    @property
    def tail_length(self):
        """Get the length of the tail section of a gene"""
        return self._head_length * (self._max_arity - 1) + 1

    @property
    def cell_length(self):
        """Get the length of the neuronal-cell section of a gene"""
        return self._head_length * (self._max_arity - 1) + 1

    @property
    def max_arity(self):
        """Get the max arity of the program-symbols"""
        return self._max_arity

    @property
    def head(self):
        """Get the head section of a gene"""
        return self[: self.head_length]

    @property
    def cell(self):
        """"Get the neuronal-cell section of a gene"""
        return self[self.head_length: self.head_length + self.cell_length]

    @property
    def tail(self):
        """Get the tail section of a gene"""
        return self[self.head_length + self.cell_length:]

    def __str__(self):
        """Get the genetic code representation of a gene"""
        rep = [str(obj) for obj in self]
        return '[{}]'.format(', '.join(rep))

    def __repr__(self):
        """Get the full string representation of a gene"""
        rep = [obj.name for obj in self]
        return '{} ['.format(self.__class__) + ', '.join(rep) + ']'


class Chromosome(entity.Chromosome):
    """Class that represents a chromosome made up of one or more genes of fixed length linear string
    consists of cellular encoding program symbols and neuronal-cell symbols.
        Note, this class extends the class *Chromosome* in the module *entity* in geppy.
        **Ref: <https://geppy.readthedocs.io/en/latest/geppy.core.html#module-geppy.core.entity>
    """
    def __init__(self, gene_generator, n_genes):
        """Instantiate a chromosome with one or more genes.
        :param gene_generator: callable, a function that generates genes, i.e. *gene_generator()*
        :param n_genes: int, number of genes that a chromosome is made of
        """
        super(Chromosome, self).__init__(gene_generator, n_genes)
        self.n_genes = n_genes

    @property
    def cell_length(self):
        """Get the cell length of a gene in a chromosome"""
        return self[0].cell_length

    @property
    def head_length(self):
        """Get the length of the head section of a gene in a chromosome"""
        return self[0].head_length

    @property
    def tail_length(self):
        """Get the length of the tail section of a gene in a chromosome"""
        return self[0].tail_length

    @property
    def linker(self):
        """Get operation that joins genes in a chromosome"""
        if self._linker is None and len(self) > 1:
            return str('concat')
        return self._linker

    @property
    def number_of_genes(self):
        return self.n_genes

    @property
    def gene_length(self):
        return len(self[0])

    def __str__(self):
        """Get the genetic code (symbols) representation of a chromosome"""
        rep = [str(obj) for obj in self]
        return '{}, linker={}'.format(',\n'.join(rep), self.linker)


class KExpressionGraph:
    """Class that represents kexpression graph, which is a mapping from genotype to phenotype. Translation
    of genotype (gene or chromosome) to a phenotype (neuronal-cell graph) by genotype-phenotype mapping.
    """
    def __init__(self, genome):
        """Instantiate a genome
        :param genome: obj, a Gene or Chromosome object
        """
        self._genome = genome

    @property
    def genome(self):
        """Get the genome of the k-expression graph"""
        return self._genome

    @classmethod
    def from_genotype(cls, genome):
        """Get the GEP-like K-expression graph entities which include list of edges and dict of
        nodes with labels. The K-expression graph is obtained by genotype-phenotype mapping.
        :param genome:obj, a Gene or Chromosome object
        :return: list, list of edges and dict of nodes with labels
        """
        if isinstance(genome, Gene):
            return [cls._kexpression_graph('0', genome)]
        if isinstance(genome, Chromosome):
            kepxr_graph = [cls._kexpression_graph(str(id), gene) for id, gene in enumerate(genome)]
            return kepxr_graph

    @classmethod
    def _kexpression_graph(cls, gene_id, genome):
        """Create a GEP-like K-expression (i.e. the open reading frame of a gene in GEP) of a genome.
        The GEP-like K-expression is a linear string form of an expression neuronal-cell graph obtained by
        genotype-phenotype mapping.
        :param id: str, gene unique id
        :param genome: obj, list containing list of edges and dict of nodes with labels
        :return: list, list of edges and dict of nodes with labels
        """
        edges_list = []
        nodes_list = {}
        kexpr_graph = []

        ps_idx = 0     # set initial program_symbol index
        cell_queue = [i for i in range(genome.head_length, len(genome))]    # create queue for the cells
        nodes_list[str(cell_queue[0]) + gene_id] = genome[cell_queue[0]].name  # add init_cell name+gene_id to nodes
        edges_list.append(str(cell_queue[0]) + gene_id)     # add init_cell index+gene_id to edges

        parent_idx = cell_queue.pop(0)   # get parent_cell index
        child_idx = parent_idx + 1       # set child_cell index

        while len(cell_queue) != 0:
            program_code = genome[ps_idx].symbol
            program_name = genome[ps_idx].name

            if program_code == 'E':
                kexpr_graph.append(nodes_list)
                kexpr_graph.append(edges_list)
                return kexpr_graph

            expr = '{}({}, {})'.format(program_name, str(parent_idx) + gene_id, str(child_idx) + gene_id)
            edges_list.append(expr)
            nodes_list[str(parent_idx) + gene_id] = genome[parent_idx].name
            nodes_list[str(child_idx) + gene_id] = genome[child_idx].name

            ps_idx += 1     # update program_symbol index
            parent_idx = cell_queue.pop(0)  # update parent_cell index
            child_idx = parent_idx + 1      # update child_cell index

        kexpr_graph.append(nodes_list)
        kexpr_graph.append(edges_list)
        return kexpr_graph


# list of classes exported
__all__ = ['Gene', 'Chromosome', 'KExpressionGraph']
