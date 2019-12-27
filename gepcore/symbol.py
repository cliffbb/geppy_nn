"""This module 'symbol' encapsulates the classes that represent the symbols in GEP-NN. A gene in
GEP-NN is made-up of multiple symbols, which include *program-symbols in cellular encoding*
    **Ref: Gruau, F. (1994) Neural Network Synthesis Using Cellular Encoding and the Genetic
           Algorithm Ph.D. Thesis, l’Ecole Normale Superieure de Lyon
    **Ref: Whitley, D., Gruau, F., and Pyeatt, L. (1995) Cellular Encoding Applied to Neurocontrol in
           the Proceedings of the Sixth International Conference (ICGA95)
And *neuronal cell symbols* that represent the basic convolution operators in CNN.
    **Ref: NASNet search space (Zoph, et. al., 2018).
Together, these symbols represent the genetic code of a gene, and they are called *primitive set*.
Some classes in this module extend the classes *Function* and *Terminal* in the module *symbol*
in geppy. **Ref: <https://geppy.readthedocs.io/en/latest/geppy.core.html#module-geppy.core.symbol>
"""
from geppy.core import symbol
from gepcore.utils.convolution import operations

def _is_char(var):
    """Check if *var* is just an alpha character"""
    if var.isalpha() and len(var) == 1:
        return True
    return False


class Symbol:
    """Class that represents a genetic code symbol"""
    def __init__(self, symbl):
        assert _is_char(symbl), "Genetic code symbol must be a single alpha character"
        self.symbl = symbl.upper()

    @property
    def symbol(self):
        return self.symbl


class ExtendFunction(symbol.Function, Symbol):
    """Class that extends *symbol.Function* in geppy to represent program-symbols in cellular encoding."""
    def __init__(self, name, symbl, arity):
        """Initialize a program-symbol in cellular encoding
        :param name: str, name of a program-symbol
        :param arity: int, arity of a program-symbol
        """
        symbol.Function.__init__(self, name, arity)
        Symbol.__init__(self, symbl)

    def __repr__(self):
        return '{}(name={}, symbol={}, arity={})'.format(self.__class__, self.name, self.symbol, self.arity)

    def __str__(self):
        return self.symbol


class ProgramSymbol(ExtendFunction):
    """Class that represents basic program-symbols in cellular encoding.
    **Ref: Gruau, F. (1994) and Frederic, D. W. et. al,. (1995)
    **Note: This class extends *Function* class in geppy
            <https://geppy.readthedocs.io/en/latest/index.html>
    """
    def __init__(self, name):
        """Initialize a basic program-symbol in cellular encoding
        :param name: str, name of a program-symbol
        """
        self.args = 2
        if name == 'par':
            self.symbl = 'P'
        elif name == 'seq':
            self.symbl = 'S'
        elif name == 'cpi':
            self.symbl = 'I'
        elif name == 'cpo':
            self.symbl = 'O'
        elif name == 'end':
            self.symbl = 'E'
            self.args = 1
        else:
            raise NotImplementedError('Unimplemented program symbol function: ', name)
        super(ProgramSymbol, self).__init__(name, symbl=self.symbl, arity=self.args)


class CellSymbol(symbol.Terminal, Symbol):
    """Class that represents *neuronal cell symbols* for convolution operations in CNN.
        **Note: This class extends *SymbolTerminal* class in geppy
        <https://geppy.readthedocs.io/en/latest/index.html>
    """
    def __init__(self, cell_symbl):
        """Initialize a neuronal cell symbol
        :param cell_symbl: tuple, a tuple with cell name and it symbol or genetic code.
        eg. ('conv3x3', 'C'), 'conv3x3' is cell name and 'C' is cell code.
        """
        if len(cell_symbl) != 2:
            raise ValueError('Cell symbol must be tuple!')

        if cell_symbl[0] not in operations:
            raise NotImplementedError('Unimplemented convolution operation: ', cell_symbl)

        symbol.Terminal.__init__(self, cell_symbl[0], value=None)
        Symbol.__init__(self, cell_symbl[1])

    def __repr__(self):
        return '{}(name={}, symbol={})'.format(self.__class__, self.name, self.symbol)

    def __str__(self):
        return self.symbol


class PrimitiveSet:
    """Class that represents primitive set of a gene, which is made up of cellular encoding
    program-symbols and neuronal cell symbols.
    """
    def __init__(self, name):
        """Initialize a gene primitive set
        :param name: str, name of a primitive set
        """
        self.codename = name
        self.terminal = []
        self.functions = []
        self.recur = []
        self.cells = []
        self.globe = {'__builtins__': None}

    def add_program_symbol(self, func):
        """Add a cellular encoding basic program-symbol to a gene primitive set.
            Note: *func* must be a predefined *gepcore.cell_graph* function for cellular encoding
                  program-symbol function. E.g. *gepcore.cell_graph.par* for parallel (PAR) program-symbol
        :param func: callable, predefined *gepcore.cell_graph* function for a program-symbol
        """
        fname = func.__name__
        self._check_uniqueness(fname, ProgramSymbol(fname).symbol)
        if fname == 'end':
            self.terminal.append(ProgramSymbol(fname))
        # elif fname == 'rec':
            # self.recur.append(ProgramSymbol(fname))
        else: self.functions.append(ProgramSymbol(fname))
        self.globe[fname] = func

    def add_cell_symbol(self, cell_symbl):
        """Add a neuronal cell symbol to a gene primitive set
        :param cell_symbl: tuple, a tuple with cell name and it symbol or genetic code.
        eg. ('conv3x3', 'C'), 'conv3x3' is cell name and 'C' is cell code.
        """
        self._check_uniqueness(cell_symbl[0], cell_symbl[1])
        self.cells.append(CellSymbol(cell_symbl))
        # self._globals[cell_symbl[0]] = cs.value

    def _check_uniqueness(self, name, symbl):
        """Check uniqueness of *name* and *symbol*"""
        symbl = symbl.upper()
        assert name not in self.globe, "Primitive with name '{}' already exists".format(name)
        assert symbl not in [obj.symbol for obj in self.functions], \
            "Primitive with genetic code '{}' already exists".format(symbl)
        assert symbl not in [obj.symbol for obj in self.terminal], \
            "Primitive with genetic code '{}' already exists".format(symbl)
        assert symbl not in [obj.symbol for obj in self.recur], \
            "Primitive with genetic code '{}' already exists".format(symbl)
        assert symbl not in [obj.symbol for obj in self.cell_symbols],\
            "Primitive with genetic code '{}' already exists".format(symbl)

    @property
    def name(self):
        """Get the code name"""
        return self.codename

    @property
    def ps_functions(self):
        """Get program-symbol functions"""
        return self.functions

    @property
    def ps_terminal(self):
        """Get program-symbol terminal"""
        return self.terminal

    @property
    def cell_symbols(self):
        """Get neuronal-cell symbols"""
        return self.cells

    @property
    def globals(self):
        """Get all program-symbol functions that can be evaluated with the builtin function *eval*."""
        return self.globe

    @property
    def max_arity(self):
        """Get the max arity of the program-symbol functions"""
        return max([obj.arity for obj in self.functions])

    def __str__(self):
        """Get an overview of the genetic codes in a gene primitive set"""
        functions = [str(obj) for obj in self.functions]
        terminal = [str(obj) for obj in self.terminal]
        recur = [str(obj) for obj in self.recur]
        cells = [str(obj) for obj in self.cell_symbols]

        return "PrimitiveSet: [{}]\n\tprogram-symbol functions: [{}]\n\tprogram-symbol terminal: [{}]" \
               "\n\tprogram-symbol recur routine: [{}]\n\tneuronal-cell symbols: [{}]".format(
                self.codename, ', '.join(functions), ', '.join(terminal), ', '.join(recur), ', '
                ''.join(cells))

# classes exported
__all__ = ['ProgramSymbol', 'CellSymbol', 'PrimitiveSet']
