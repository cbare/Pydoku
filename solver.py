# Sudoku solver
# Sept. 2011
# J. Christopher Bare

# note: How this will work on sudoku's other than 9x9 is untested. It
# won't work on underspecified puzzles - those with more than one solution.
# Is it possible to have puzzles with a unique solution, but that requires
# you to make a guess and back-track if you're wrong?

import sys
import cStringIO
from math import sqrt, ceil
import argparse
from itertools import chain


class Square(object):
    """Represent one square of a sudoku puzzle. The square is at position
    i,j on the board. Square.value will be 0 if the value is unknown and
    possible values will be in Square.values.
    """
    def __init__(self, i, j, value=0):
        self.i = i
        self.j = j
        if value==0:
            self.values = {1,2,3,4,5,6,7,8,9}
        else:
            self.values = {value}

    def __str__(self):
        if self.solved():
            return "square(%d,%d) = %d" % (self.i, self.j, self.value)
        else:
            return "square(%d,%d) = {%s}" % (self.i, self.j, ",".join([str(n) for n in self.values]))

    def __repr__(self):
        return self.__str__()

    @property
    def value(self):
        """
        The value in the square or '-' if the value is not yet determined
        """
        if len(self.values) == 1:
            return list(self.values)[0]
        else:
            return '-'

    @value.setter
    def value(self, value):
        self.values = {value}

    def set_possible_values(self, values):
        self.values = values

    def eliminate(self, values):
        eliminated = self.values.intersection(values)
        self.values -= values
        return eliminated

    def npossibilities(self):
        """Return the number of possible values for this square."""
        return len(self.values)

    def solved(self):
        return len(self.values)==1

    def unsolved(self):
        return len(self.values) > 1


class Sudoku(object):
    """Represent an n by m sudoku board."""
    def __init__(self, n=9, m=9, board=[]):
        self.n = n
        self.m = m
        self.board = board
        self.verbose = False
    
    def get_square(self, i, j):
        return self.board[ i*self.m + j ]

    def solved(self):
        for square in self.board:
            if not square.solved():
                return False
        return True

    ## instead we could have a row, column and box function and a generalized
    ## neighbor function that checks and doesn't return the original square

    def row_neighbors(self, square):
        """Generate the squares in the same row as the given square"""
        for j in range(0, self.m):
            if j!=square.j:
                yield self.get_square(square.i, j)

    def column_neighbors(self, square):
        """Generate the squares in the same column as the given square"""
        for i in range(0, self.n):
            if i!=square.i:
                yield self.get_square(i, square.j)

    def box_neighbors(self, square):
        """Generate the squares in the same box as the given square"""
        box_i = square.i / 3 * 3
        box_j = square.j / 3 * 3
        for i in range(box_i, box_i+3):
            for j in range(box_j, box_j+3):
                if (not (i==square.i and j==square.j)):
                    yield self.get_square(i, j)
    
    def rows(self):
        for i in range(0, self.n):
            row = []
            for j in range(0, self.m):
                row.append(self.get_square(i,j))
            yield row

    def columns(self):
        for j in range(0, self.m):
            column = []
            for i in range(0, self.n):
                column.append(self.get_square(i,j))
            yield column

    def boxes(self):
        for bi in range(0, int(sqrt(self.n))):
            for bj in range(0, int(sqrt(self.m))):
                box = []
                for i in range(0, int(sqrt(self.n))):
                    for j in range(0, int(sqrt(self.m))):
                        box.append(self.get_square(bi*int(sqrt(self.n))+i, bj*int(sqrt(self.m))+j))
                yield box

    def check_solution(self):
        """Check that our solution really has the sudoku properties"""
        if not self.solved():
            return False
        return all([
                all([
                    all([i in values_of(squares)
                        for i in range(1,self.n+1)])
                            for squares in f()])
                                for f in [self.rows, self.columns, self.boxes]])

    def count_solved_squares(self):
        """How many solved squares are in the puzzle?"""
        count = 0
        for square in self.board:
            if square.solved():
                count += 1
        return count

    def __str__(self):
        """returns ascii art rendering of sudoku"""
        box_size = int(sqrt(self.n))
        try:
            output = cStringIO.StringIO()
            for i in range(0, self.n):
                if i > 0 and i % box_size == 0:
                    for j in range(1,box_size):
                        output.write('-' * (box_size*2 + 1) + "+")
                    output.write('-' * (box_size*2 + 1) + "\n")
                row_values = []
                for j in range(0, self.m):
                    if j > 0 and j % box_size == 0:
                        output.write(' |')
                    output.write(' ')
                    output.write( str(self.get_square(i,j).value) )
                output.write("\n")
            return output.getvalue()
        finally:
            output.close()


    def details(self):
        """returns ascii art rendering of sudoku"""
        box_size = int(sqrt(self.n))
        try:
            output = cStringIO.StringIO()
            for i in range(self.n):
                output.write('\n')
                if i > 0 and i % box_size == 0:
                    output.write('-' * (self.n*self.n + self.n-1))
                    output.write('\n')
                for j in range(self.m):
                    if j > 0:
                        if j % box_size == 0:
                            output.write('|')
                        else:
                            output.write(' ')
                    values = self.get_square(i,j).values
                    output.write(' ' * ((self.n - len(values))/2))
                    output.write(''.join([str(value) for value in values]))
                    output.write(' ' * (int(ceil((self.n - len(values))/2.0))))
            output.write('\n')
            return output.getvalue()
        finally:
            output.close()



def read_sudoku_from_file(filename):
    """Read a sudoku puzzle from a text file."""
    board = []
    row = 0
    ncol = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith("#"): continue
            line = line.strip("\n")
            if line:
                ncol = len(line)
                for col in range(0,ncol):
                    if (line[col].isdigit()):
                        board.append(Square(row, col, value=int(line[col])))
                    else:
                        board.append(Square(row, col))
                row += 1
    return Sudoku(row,ncol,board)




from collections import namedtuple

Application = namedtuple('Application', ['rule', 'square', 'new_values'])


def solved(squares):
    for square in squares:
        if len(square.values)==1:
            yield square

def unsolved(squares):
    for square in squares:
        if len(square.values) > 1:
            yield square

def values_of(squares):
    values = set()
    for square in squares:
        values = values.union(square.values)
    return values


def subsets(s):
    """Given a set s, return all subsets of s"""
    if len(s)==0:
        return [()]
    rest = list(s)
    first = rest.pop()
    results = subsets(rest)
    results.extend([(first,) + s for s in results])
    return results

def subsets_of_limited_cardinality(s, n, m):
    """Given a set s, generate all subsets of s whose cardinality is at least n and < m"""
    ## It would be cooler to construct these subsets directly, but here we
    ## generate all subsets then filter for the those of requested cardinality.
    return (subset for subset in subsets(s) if len(subset) >= n and len(subset) < m)


## RULES
##
## Elimination by itself is sufficient for easy puzzles. Adding
## deduction will find solutions for medium and even hard puzzles.
## ------------------------------------------------------------------

def eliminate_by(neighbors, description):
    """
    Return a function of the form `f(sudoku, square)` that eliminates from
    `square` the values of all solved squares returned by the neighbors
    function, which may be Sudoku.row_neighbors, Sudoku.column_neighbors or
    Sudoku.box_neighbors.

    The returned function takes a sudoku and a specific square and returns an
    Application object, which may then be applied to reduce the possible values
    of the square.

    example::

        eliminate_by_row = eliminate_by(Sudoku.row_neighbors, description='by solved row neighbors')
        application = eliminate_by_row(sudoku, square)

    """
    def eliminate(sudoku, history=[]):
        """
        Eliminate the values of solved neighboring squares from the possible values of
        the square under consideration.
        """
        for square in unsolved(sudoku.board):
            values = values_of(solved(neighbors(sudoku, square)))
            values_to_eliminate = square.values.intersection(values)
            if len(values_to_eliminate) > 0:
                history.append(
                    Application(
                        "Eliminated {values} from {square} {description}"
                            .format(
                                values=','.join([str(i) for i in values_to_eliminate]),
                                square=','.join([str(square.i), str(square.j)]),
                                description=description
                            ),
                       square,
                       square.values - values_to_eliminate))
                square.eliminate(values_to_eliminate)
        return history
    return eliminate

def deduce_by(neighbors, description):
    def deduce(sudoku, history=[]):
        for square in unsolved(sudoku.board):
            values = values_of(neighbors(sudoku, square))
            values_only_possible_in_this_square = square.values - values
            if len(values_only_possible_in_this_square)==1:
                history.append(
                    Application(
                        "Deduced {square} holds {value} {description}"
                            .format(
                                square=','.join([str(square.i), str(square.j)]),
                                value=list(values_only_possible_in_this_square)[0],
                                description=description
                            ),
                        square,
                        values_only_possible_in_this_square))
                square.set_possible_values(values_only_possible_in_this_square)
        return history
    return deduce

    
def box_row_intersection(sudoku, history=[]):
    """If all squares in a box that could hold a number n are in the same row
    or column, we can eliminate that n from the rest of that row or column"""
    for row in chain(sudoku.rows(), sudoku.columns()):
        for box in sudoku.boxes():
            # take union of possibilities in of the squares in (row INTERSECT box)
            # subtract from that the union of possibilities of the squares in (box - row)
            # eliminate those possibilities from all squares in (row - box)
            a = set()
            for square in (set(row) & set(box)):
                a.update(square.values)
            if len(a) == 0: continue
            b = set()
            for square in (set(box) - set(row)):
                b.update(square.values)
            c = a - b
            if len(c) == 0: continue
            for square in (set(row) - set(box)):
                e = square.eliminate(c)
                if e:
                    history.append(
                        Application(
                            "Eliminated {values} from {square} by box-row intersection"
                                .format(
                                    square=','.join([str(square.i), str(square.j)]),
                                    values=','.join([str(i) for i in e])
                                ),
                            square,
                            square.values))
    return history

def row_box_intersection(sudoku, history=[]):
    for row in chain(sudoku.rows(), sudoku.columns()):
        for box in sudoku.boxes():
            a = values_of(set(row) & set(box))
            b = values_of(set(row) - set(box))
            c = a - b
            if len(c) > 0:
                for square in (set(box) - set(row)):
                    e = square.eliminate(c)
                    if e:
                        history.append(
                            Application(
                                "Eliminated {values} from {square} by row-box intersection"
                                    .format(
                                        square=','.join([str(square.i), str(square.j)]),
                                        values=','.join([str(i) for i in e])
                                    ),
                                square,
                                square.values))
    return history

def closed_subsets_by(containers, description):
    def closed_subsets(sudoku, history=[]):
        for container in containers(sudoku):
            unsolved_squares = list(unsolved(container))
            for subset in subsets_of_limited_cardinality(unsolved_squares, 2, len(unsolved_squares)):
                possibilities = values_of(subset)
                if len(possibilities) == len(subset):
                    for square in unsolved_squares:
                        if square not in subset:
                            eliminated = square.eliminate(possibilities)
                            if eliminated:
                                history.append(Application(
                                    "Eliminated {values} from {square} by closed subset {description}"
                                        .format(
                                            values=",".join([str(e) for e in eliminated]),
                                            square=','.join([str(square.i), str(square.j)]),
                                            description=description
                                        ), square, square.values))
        return history
    return closed_subsets

rules = []
rules.append(eliminate_by(Sudoku.row_neighbors, description='by solved row neighbors'))
rules.append(eliminate_by(Sudoku.column_neighbors, description='by solved column neighbors'))
rules.append(eliminate_by(Sudoku.box_neighbors, description='by solved box neighbors'))
rules.append(deduce_by(Sudoku.row_neighbors, description='by row'))
rules.append(deduce_by(Sudoku.column_neighbors, description='by column'))
rules.append(deduce_by(Sudoku.box_neighbors, description='by box'))
rules.append(box_row_intersection)
rules.append(row_box_intersection)
rules.append(closed_subsets_by(Sudoku.rows, description='in row'))
rules.append(closed_subsets_by(Sudoku.columns, description='in column'))
rules.append(closed_subsets_by(Sudoku.boxes, description='in box'))

## ------------------------------------------------------------


def apply_rules(sudoku, rules, history=[], verbose=False):
    ## while not solved and we're still making progress, keep applying rules
    i = 1
    while not sudoku.solved():
        progress = []
        for rule in rules:
            rule(sudoku, progress)
        if progress:
            history.append(progress)
            print('\n\niteration %d' % i)
            print('-' * 80)
            for application in progress:
                print application.rule, application.square, application.new_values
            print sudoku.details()
        else:
            print('\n\n' + '-'*60)
            print("...uh-oh, not making progress!")
            break
        i += 1
    return history


class Solver(object):
    """
    A solver consists of a set of rules.
    """
    def __init__(self, rules=[]):
        self.rules=rules
        self.history = None

    def add_rule(self, rule):
        self.rules.append(rule)

    def solve(self, sudoku):
        self.history = apply_rules(sudoku, self.rules)
        if sudoku.check_solution():
            print("Found a solution!")
        return self.history


def main():
    # handle command line arguments
    parser = argparse.ArgumentParser(
        description="cbare's sudoku solver.",
        epilog="example: python solver.py sudoku.txt")
    parser.add_argument("-v", "--verbose",
        action='store_true', default=False,
        help="show solution for each square.")
    parser.add_argument("filename")
    args = parser.parse_args()

    # Read a sudoku puzzle from a text file and solve it.
    sudoku = read_sudoku_from_file(args.filename)

    solver = Solver(rules)

    print("\n%d squares given.\n" % (sudoku.count_solved_squares()))
    print(sudoku)
    solver.solve(sudoku)
    print
    print(sudoku)


if __name__ == "__main__":
    main()
