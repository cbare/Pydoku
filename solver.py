# Sudoku solver
# Sept. 2011
# J. Christopher Bare

# note: How this will work on sudoku's other than 9x9 is untested. It
# won't work on underspecified puzzles - those with more than one solution.
# Is it possible to have puzzles with a unique solution, but that requires
# you to make a guess and back-track if you're wrong?

import sys
import cStringIO
from math import sqrt
import argparse
from itertools import chain


def sort_unique(l):
    result = list(set(l))
    result.sort()
    return result

class Square:
    """Represent one square of a sudoku puzzle. The square is at position
    i,j on the board. Square.value will be 0 if the value is unknown and
    possible values will be in Square.values.
    """
    def __init__(self, i, j, value=0):
        self.i = i
        self.j = j
        self.value = value
        if value > 0:
            self.values = {value}
        else:
            self.values = {1,2,3,4,5,6,7,8,9}

    def __str__(self):
        if self.solved():
            return "square(%d,%d) = %d" % (self.i, self.j, self.value)
        else:
            return "square(%d,%d) = {%s}" % (self.i, self.j, ",".join([str(n) for n in self.values]))

    def eliminate(self, n):
        """Take either an integer or a list. Eliminate those values from
        the possible values for this square. It there is only one possibility
        remaining, the square is solved and square.value will be assigned.
        """
        result = []
        if (type(n)==list or type(n)==set):
            for nn in n:
                if (nn in self.values):
                    self.values.remove(nn)
                    result.append(nn)
        else:
            if (n in self.values):
                self.values.remove(n)
                result.append(n)

        if len(self.values)==1:
            self.value = list(self.values)[0]

        return result
    
    def set_value(self, value):
        self.value = value
        self.values = {value}

    def set_possible_values(self, values):
        self.values = values
        if len(values)==1:
            self.value = list(values)[0]

    def npossibilities(self):
        """Return the number of possible values for this square."""
        return len(self.values)

    def solved(self):
        return len(self.values)==1

    def unsolved(self):
        return len(self.values) > 1


class Sudoku:
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

    def solve(self):
        unsolved = 0
        for square in self.board:
            if not square.solved():
                unsolved += 1

        iteration = 0
        while (unsolved > 0):
            iteration += 1
            progress = 0
            if self.verbose : print "iteration %d, %d unsolved squares remaining." % (iteration, unsolved)
            for square in self.board:
                if not square.solved():
                    
                    if self.verbose : print "  " + str(square)
                    
                    print str(self)
                    
                    # eliminate solved numbers in row, column, box
                    eliminated = []
                    eliminated.extend( self.eliminate(square, self.row_neighbors(square)) )
                    eliminated.extend( self.eliminate(square, self.column_neighbors(square)) )
                    eliminated.extend( self.eliminate(square, self.box_neighbors(square)) )
                    
                    if self.verbose  and len(eliminated) > 0:
                        print "    eliminated: {%s}" % (",".join([str(n) for n in sort_unique(eliminated)],))
                        if square.unsolved():
                            print "    " + str(square)
                        progress += 1
                    
                    # deduce that this square is the only square in a row,
                    # column or box that could hold a particular number
                    self.deduce(square, self.row_neighbors(square))
                    self.deduce(square, self.column_neighbors(square))
                    self.deduce(square, self.box_neighbors(square))
                    
                    if square.solved():
                        unsolved -= 1
                        progress += 1
                        if self.verbose : print "    >>>> solved: " + str(square)
                    elif len(square.values)==0:
                        print "warning: square with no possible values! " + str(square)

            # TODO properly mark solved squares in closed subsets methods
            # TODO implement rule of 3
            # TODO keep track of derivation for each square
            
            for row in self.rows():
                progress += self.eliminate_by_closed_subsets(row)
            for column in self.columns():
                progress += self.eliminate_by_closed_subsets(column)
            for box in self.boxes():
                progress += self.eliminate_by_closed_subsets(box)
            
            progress += len(self.eliminate_by_intersection())

            # we should always make progress. If not, something's wrong.
            if (progress == 0):
                print("...uh-oh, not making progress!")
                for square in self.board:
                    print square
                break

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


    def eliminate(self, square, neighbors):
        """Every row, column and box must have at most one of each
        number, so eliminate any number that's already solved in a
        neighboring square in the same row, column, box.
        """
        eliminated = []
        for neighbor in neighbors:
            if neighbor.solved():
                eliminated.extend( square.eliminate(neighbor.value) )
        return eliminated

    def deduce(self, square, neigbors):
        """Every row, column and box must have at least one of each
        number, so if this square might hold a number that no
        neighboring square could hold, deduce that this square does
        hold that number.
        """
        if not square.solved():
            original_possibilities = square.values
            neighbors_possibilities = set()
            possibilities = square.values
            for neighbor in neigbors:
                if neighbor.unsolved():
                    possibilities = possibilities - neighbor.values
                    neighbors_possibilities.update(neighbor.values)
            if len(possibilities) == 1:
                if self.verbose :
                    print "    {%s} - {%s} = {%s}" % (
                        ",".join([str(n) for n in original_possibilities]),
                        ",".join([str(n) for n in neighbors_possibilities]),
                        ",".join([str(n) for n in possibilities]))
                square.set_value( list(possibilities)[0] )
    
    def eliminate_by_intersection(self):
        """If all squares in a box that could hold a number n are in the same row
        or column, we can eliminate that n from the rest of that row or column"""
        eliminated = []
        for row in chain(self.rows(), self.columns()):
            for box in self.boxes():
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
                    if len(e) > 0:
                        print "@@@ eliminated " + str(eliminated)
                        eliminated.extend(e)
        return eliminated
    
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

    def subsets(self, squares):
        if len(squares)==0:
            return [[]]
        rest = []
        rest.extend(squares[1:])
        results = self.subsets(rest)
        results.extend([[squares[0]] + s for s in results])
        return results

    # a closed set is my made-up term for a set of n squares
    # that together can hold n possible values. Therefore, it's
    # certain that the those squares contain those values in some
    # order and we can eliminate those values from the remainder of
    # the row, column or box.
    # returns progress, if any or 0 to represent no progress
    def eliminate_by_closed_subsets(self, squares):
        progress = 0
        unsolved_squares = [ square for square in squares if square.unsolved() ]
        if len(unsolved_squares) <= 2: return progress
        good_subsets = [ subset for subset
            in self.subsets(unsolved_squares)
            if len(subset) > 1 and len(subset) < len(unsolved_squares) ]
        for subset in good_subsets:
            possibilities = set()
            for square in subset:
                possibilities.update(square.values)
            if len(possibilities) == len(subset):
                if self.verbose: print("  eliminating {%s} by closed subsets from: " % (",".join([str(p) for p in possibilities])))
                for s1 in subset:
                    print "---> " + str(s1)
                for square in unsolved_squares:
                    if square not in subset:
                        if self.verbose: print "    " + str(square)
                        progress += len(square.eliminate(possibilities))
        return progress

    # todo
    def check_solution(self):
        """Check that our solution really has the sudoku properties"""
        pass

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


    def str2(self):
        """returns ascii art rendering of sudoku"""
        box_size = int(sqrt(self.n))
        try:
            output = cStringIO.StringIO()

            for i in range(0,box_size):
                for j in range(0,box_size):
                    output.write("%d,%d" % (i, j))
                output.write("+".join( ['-' * box_size] * box_size ))
                output.write("\n")

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
    values = set([])
    for square in squares:
        values = values.union(square.values)
    return values


def subsets(s):
    """Given a set s, return all proper subsets of s"""
    if len(s)==0:
        return [[]]
    rest = s[1:]
    results = subsets(rest)
    results.extend([[s[0]] + s for s in results])
    return results


def apply_rule(sudoku, rule, history=[]):
    for square in unsolved(sudoku.board):
        application = rule(sudoku, square)
        if application:
            square.set_possible_values(application.new_values)
            history.append(application)
    return history


def apply_rules(sudoku, rules, history=[]):
    ## while not solved and we're still making progress, keep applying rules
    i = 1
    while not sudoku.solved():
        progress = []
        for rule in rules:
            apply_rule(sudoku, rule, progress)
        if progress:
            history.append(progress)
            print '\n\niteration %d' % i
            print '-' * 80
            for application in progress:
                print application
        else:
            break
        i += 1
    return history


## RULES
## ------------------------------------------------------------

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
    def eliminate(sudoku, square):
        """
        Take a sudoku and a sqaure. Return an Application object that
        eliminates some possible values from the square.
        """
        values = values_of(solved(neighbors(sudoku, square)))
        values_to_eliminate = square.values.intersection(values)
        if len(values_to_eliminate) > 0:
            return Application(
                "Eliminated {values} from {square} {description}"
                    .format(
                        values=','.join([str(i) for i in values]),
                        square=','.join([str(square.i), str(square.j)]),
                        description=description
                    ),
               square,
               square.values - values_to_eliminate)
    return eliminate

def deduce_by(neighbors, description):
    def deduce(sudoku, square):
        values = values_of(neighbors(sudoku, square))
        values_only_possible_in_this_square = square.values - values
        if len(values_only_possible_in_this_square)==1:
            return Application(
                "Deduced {square} holds {value} {description}"
                    .format(
                        square=','.join([str(square.i), str(square.j)]),
                        value=list(values_only_possible_in_this_square)[0],
                        description=description
                    ),
                square,
                values_only_possible_in_this_square)
        return None
    return deduce

## ------------------------------------------------------------


def solve(sudoku):
    rules = []
    rules.append(eliminate_by(Sudoku.row_neighbors, description='by solved row neighbors'))
    rules.append(eliminate_by(Sudoku.column_neighbors, description='by solved column neighbors'))
    rules.append(eliminate_by(Sudoku.box_neighbors, description='by box column neighbors'))
    rules.append(deduce_by(Sudoku.row_neighbors, description='by row'))
    rules.append(deduce_by(Sudoku.column_neighbors, description='by column'))
    rules.append(deduce_by(Sudoku.box_neighbors, description='by box'))

    history = apply_rules(sudoku, rules)


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

    print("\n%d squares given.\n" % (sudoku.count_solved_squares()))
    print(sudoku)
    sudoku.solve()
    print
    print(sudoku)


if __name__ == "__main__":
    main()