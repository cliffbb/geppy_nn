"""This module 'ea_operations' encapsulates the necessary evolutionary algorithm (EA) operations for
genetic modifications in GEP, including one/two-point mutation, inversion mutation, transpose
mutation, and recombination between multi-genetic chromosomes.
    **Ref: Ferreira, Cândida. Gene expression programming: mathematical modeling by an artificial
           intelligence. Vol. 21. Springer, 2006 (Chapter 3)
"""
import random

_DEBUG = False


def select_cell(pset):
    return random.choice(pset.cell_symbols)


def select_program(pset):
    return random.choice(pset.ps_functions + pset.ps_terminal)


def cross_one_point(indv1, indv2):
    """Get one-point recombination of two individuals/chromosomes. The operation is performed in
    place, and the two children/chromosomes are returned.
    :param indv1: obj, the first individual as parent_1 in the crossover operation
    :param indv2: obj, the second individual as parent_2 in the crossover operation
    :return: tuple, the two offspring of the parents
    """
    assert len(indv1) == len(indv2), 'Chromosomes must have same # of genes'
    gene_size = len(indv1[0])
    indv_size = len(indv1)
    if indv_size > 1:
        gene = random.randint(0, indv_size - 1)
        cx_pt = random.randint(0, gene_size - 1)
        indv1[gene:], indv2[gene:] = indv2[gene:], indv1[gene:]
        indv1[gene][cx_pt:], indv2[gene][cx_pt:] = indv2[gene][cx_pt:], indv1[gene][cx_pt:]
        if _DEBUG: print('Gene {} crossed at point {}'.format(gene, cx_pt))
    elif indv_size == 1:
        cx_pt = random.randint(0, gene_size - 1)
        indv1[0][cx_pt:], indv2[0][cx_pt:] = indv2[0][cx_pt:], indv1[0][cx_pt:]
        if _DEBUG: print('Genes crossed at point {}'.format(cx_pt))
    return indv1, indv2


def cross_two_point(indv1, indv2):
    """Get two-point recombination of two individuals/chromosomes. The operation is performed in
    place, and the two children/chromosomes are returned.
    :param indv1: obj, the first individual as parent_1 in the crossover operation
    :param indv2: obj, the second individual as parent_2 in the crossover operation
    :return: tuple, the two offspring of the parents
    """
    assert len(indv1) == len(indv2), 'Chromosomes must have same # of genes'
    gene_size = len(indv1[0])
    indv_size = len(indv1)
    if indv_size > 1:
        gene1 = random.randint(0, indv_size - 2)
        gene2 = random.randint(0, indv_size - 1)
        g1_cx_pt = random.randint(0, gene_size - 2)
        g2_cx_pt = random.randint(0, gene_size - 1)
        if gene1 > gene2:
            gene1, gene2 = gene2, gene1
        elif gene1 == gene2:
            gene2 += 1
        if g1_cx_pt > g2_cx_pt:
            g1_cx_pt, g2_cx_pt = g2_cx_pt, g1_cx_pt
        elif g1_cx_pt == g2_cx_pt:
            g2_cx_pt += 1
        indv1[gene1:gene2], indv2[gene1:gene2] = indv2[gene1:gene2], indv1[gene1:gene2]
        indv1[gene1][g1_cx_pt:], indv2[gene1][g1_cx_pt:] = indv2[gene1][g1_cx_pt:], indv1[gene1][g1_cx_pt:]
        indv1[gene2][:g2_cx_pt + 1], indv2[gene2][:g2_cx_pt + 1] = \
            indv2[gene2][:g2_cx_pt + 1], indv1[gene2][:g2_cx_pt + 1]
        if _DEBUG: print('Gene {} and Gene {} crossed at points {} and {} respectively'
                  .format(gene1, gene2, g1_cx_pt, g2_cx_pt))
    elif indv_size == 1:
        cx_pt1 = random.randint(0, gene_size - 2)
        cx_pt2 = random.randint(0, gene_size - 1)
        if cx_pt1 > cx_pt2:
            cx_pt1, cx_pt2 = cx_pt2, cx_pt1
        elif cx_pt1 == cx_pt2:
            cx_pt2 += 1
        indv1[0][cx_pt1:cx_pt2 + 1], indv2[0][cx_pt1:cx_pt2 + 1] = \
            indv2[0][cx_pt1:cx_pt2 + 1], indv1[0][cx_pt1:cx_pt2 + 1]
        if _DEBUG: print('Genes crossed at points {} and {} '.format(cx_pt1, cx_pt2))
    return indv1, indv2


def cross_gene(indv1, indv2):
    """Get gene recombination of two individuals/chromosomes. An entire gene is exchanged between two
    individual parents. The operation is performed in place, and the two children/chromosomes
    are returned.
    :param indv1: obj, the first individual as parent_1 in the crossover operation
    :param indv2: obj, the second individual as parent_2 in the crossover operation
    :return: tuple, the two offspring of the parents
    """
    assert len(indv1) == len(indv2), 'Chromosomes must have same # of genes'
    length = len(indv1)
    if length > 1:
        cx_gene = random.randint(0, length - 1)
        indv1[cx_gene], indv2[cx_gene] = indv2[cx_gene], indv1[cx_gene]
        if _DEBUG: print('Genes at position {} crossed'.format(cx_gene))
    else:
        raise ValueError('Chromosome has only one gene')
    return indv1, indv2


def mutate_uniform(indv, pset, indpb=None):
    """Get uniform mutation of each gene in a chromosome. Each gene's primitive symbols in
    an individual (chromosome) is changed to a randomly selected primitive symbol with a given
    probability (mutation rate). Typically, primitive symbol mutation rate 'indpb' is
    equivalent to two one-point mutations per chromosome. That is, indpb = 2 / len(chromosome) * len(gene).
    :param indv: obj, the individual/chromosome to be mutated
    :param pset: obj, the primitive set (gene codes) of the individual
    :param indpb: float,
    :return: tuple, one child of an individual parent
    """
    if indpb is None:
        indpb = 2 / (len(indv) * len(indv[0]))
    head = indv[0].head_length
    cell = indv[0].cell_length
    for i, gene in enumerate(indv):
        for j in range(head):
            if random.random() < indpb:
                gene[j] = select_program(pset)
                if _DEBUG: print('Gene {} mutated at point {}'.format(i, j))
        for k in range(head, head + cell):
            if random.random() < indpb:
                gene[k] = select_cell(pset)
                if _DEBUG: print('Gene {} mutated at point {}'.format(i, k))
    return indv,


def invert_program(indv, size=2):
    """Get program-symbol inversion mutation of a chromosome with a given probability. A gene in a
    chromosome is randomly selected, and sequence of program-symbols within the gene is randomly selected and inverted.
    Typically, inversion rate is set to 0.1
    :param indv: obj, the individual/chromosome to be mutated
    :param size: int, the length of string to invert
    :return: tuple, one child of an individual parent
    """
    head = indv[0].head_length
    if head < size:
        return indv,

    length = len(indv)
    if length > 1:
        idx = random.randint(0, length - 1)
        gene = indv[idx]
        stpt = random.randint(0, head - size)
        endpt = stpt + size
        gene[stpt:endpt] = reversed(gene[stpt:endpt])
        if _DEBUG: print('{} program-symbols of gene {} are inverted at positions [{} - {}]'
                  .format(size, idx, stpt, endpt - 1))
    elif length == 1:
        stpt = random.randint(0, head - size)
        endpt = stpt + size
        indv[0][stpt:endpt] = reversed(indv[0][stpt:endpt])
        if _DEBUG: print('{} program-symbols have been inverted at positions [{} - {}]'
                         .format(size, stpt, endpt - 1))
    return indv,


def invert_cell(indv, size=2):
    """Get cell inversion mutation of a chromosome with a given probability. A gene in a
    chromosome is randomly selected, and sequence of cells within the gene is randomly selected and inverted.
    Typically, inversion rate is set to 0.1
    :param indv: obj, the individual/chromosome to be mutated
    :param size: int, the length of string to invert
    :return: tuple, one child of an individual parent
    """
    cell = indv[0].cell_length
    if cell < size:
        return indv,

    length = len(indv)
    head = indv[0].head_length
    if length > 1:
        idx = random.randint(0, length - 1)
        gene = indv[idx]
        stpt = random.randint(head, head + cell - size )
        endpt = stpt + size
        gene[stpt:endpt] = reversed(gene[stpt:endpt])
        if _DEBUG: print('{} cells of gene {} are inverted at positions [{} - {}]'
                         .format(size, idx, stpt, endpt - 1))
    elif length == 1:
        stpt = random.randint(head, head + cell - size)
        endpt = stpt + size
        indv[0][stpt:endpt] = reversed(indv[0][stpt:endpt])
        if _DEBUG: print('{} cells have been inverted at positions [{} - {}]'.format(size, stpt, endpt - 1))
    return indv,


def transpose_program(indv, size=2):
    """Get program insertion-sequence transposition mutation of a chromosome with a given probability.
    A gene in a chromosome is randomly selected, and a sequence of program-symbols is randomly selected,
    and a copy is afterwards inserted at a randomly selected point in the head of another gene in the chromosome.
    Typically, transposition rate is set to 0.1
    :param indv: obj, the individual/chromosome to be mutated
    :param size: int, the length of string to transpose
    :return tuple, one child of an individual parent
    """
    length = len(indv)
    head = indv[0].head_length
    if length == 1 or head < size:
        return indv,

    gene1 = random.randint(0, length - 2)
    gene2 = random.randint(0, length - 1)
    if gene1 > gene2:
        gene1, gene2 = gene2, gene1
    elif gene1 == gene2:
        gene2 += 1
    stpt1 = random.randint(0, head - size)
    stpt2 = random.randint(0, head - size)
    endpt1 = stpt1 + size
    endpt2 = stpt2 + size
    donner, target = indv[gene1], indv[gene2]
    target[stpt2:endpt2] = donner[stpt1:endpt1]
    if _DEBUG: print('Gene {} transposed program-symbols at positions [{} - {}] to Gene {} at positions '
                     '[{} - {}]'.format(gene1, stpt1, endpt1 - 1, gene2, stpt2, endpt2 - 1))
    return indv,


def transpose_cell(indv, size=2):
    """Get cell insertion-sequence transposition mutation of a chromosome with a given probability.
    A gene in a chromosome is randomly selected, and a sequence of cells is randomly selected, and a copy
    is afterwards inserted at a randomly selected point in the cell domain of another gene in the chromosome.
    Typically, transposition rate is set to 0.1
    :param indv: obj, the individual/chromosome to be mutated
    :param size: int, the length of string to transpose
    :return: tuple, one child of an individual parent
    """
    length = len(indv)
    head = indv[0].head_length
    cell = indv[0].cell_length
    if length == 1 or cell < size:
        return indv,

    gene1 = random.randint(0, length - 2)
    gene2 = random.randint(0, length - 1)
    if gene1 > gene2:
        gene1, gene2 = gene2, gene1
    elif gene1 == gene2:
        gene2 += 1
    stpt1 = random.randint(head, head + cell - size)
    stpt2 = random.randint(head, head + cell - size)
    endpt1 = stpt1 + size
    endpt2 = stpt2 + size
    donner, target = indv[gene1], indv[gene2]
    target[stpt2:endpt2] = donner[stpt1:endpt1]
    if _DEBUG: print('Gene {} transposed cells at positions [{} - {}] to Gene {} at positions [{} - {}]'
              .format(gene1, stpt1, stpt1 - 1, gene2, stpt2, endpt2 - 1))
    return indv,


def transpose_gene(indv):
    """Get gene transposition mutation of a chromosome with a given probability. A gene in a
    chromosome is randomly selected, and the entire gene is transposed to the beginning of the
    chromosome. To maintain the length of the chromosome, the transposon (the gene which was
    transposed) is deleted at its place of origin. Typically, transposition rate is set to 0.1
    :param indv: obj, the individual/chromosome to be mutated
    :return: tuple, one child of an individual parent
    """
    if len(indv) == 1:
        return indv,

    idx = random.randint(1, len(indv) - 1)
    gene = indv[idx]
    indv.remove(gene)
    indv.insert(0, gene)
    if _DEBUG: print('Gene {} has been transposed to position 0'.format(idx))
    return indv,

# exported functions
__all__ = ['cross_one_point', 'cross_two_point', 'cross_gene', 'mutate_uniform', 'invert_program',
           'invert_cell', 'transpose_cell', 'transpose_program', 'transpose_gene']
