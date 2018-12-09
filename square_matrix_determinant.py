"""
    For given square matrix find its determinant.

    In the first row inputs integer value 'n' (1≤n≤12).
    Next 'n' rows contains n^2 integer values from -5 to 5 ('n' number in each row)
                        (elements of the matrix).

    Print one integer value - determinant of the given matrix.
"""


class SquareMatrix():
    _min_value = -5
    _max_value = 5
    _max_size = 12

    def __init__(self, size):
        """ Square matrix initialization with zeros;
            size = i(rows) = j(columns), 1 <= size <= 12
        """

        assert (1 <= size <= self._max_size) # check for matrix size
        self.size = size

        self.matrix = []
        for row in range(size):
            self.matrix.append([0 for column in range(size)])

    def fill_from(self, list_of_lists=[[]]):
        for i, row in enumerate(list_of_lists):
            if i >= self.size:
                break

            for j, element in enumerate(row):
                if j >= self.size:
                    break

                assert (self._min_value <= element <= self._max_value)
                self.matrix[i][j] = element

    def print(self, matrix=None):
        if matrix is None:
            matrix = self.matrix

        for row in matrix:
            for element in row:
                print("{0:^4}".format(str(element)[:4]), end=' ')
            print()

    def __str__(self):
        # if 'A = SquareMatrix()' ==> print(A)
        size = self.get_size()
        print("\nIntance of '{}', size = {}x{}".format(self.__class__.__name__, size, size))
        self.print()
        return ''

    def determinant(self, matrix=None, sign=1):
        """ recursive determinant calculation """

        if matrix is None:
            # first iteration
            return self.determinant(matrix=self.matrix)

        assert (matrix != []) & (matrix != [[]])

        size = self.get_size(matrix=matrix)
        if size == 1:
            # determinant of 1-element matrix is this element
            return sign * matrix[0][0]

        if self.is_null_column(column=1, matrix=matrix):
            # if first column consints only of 0 then determinant = 0
            return 0

        edited_matrix = self.copy(matrix=matrix)

        if edited_matrix[0][0] == 0:
            # swaping first row with the other one, which starts not from 0
            for i in range(1, size):
                if edited_matrix[i][0] != 0:
                    self.swap_rows(1, i+1, matrix=edited_matrix)
                    sign *= -1
                    break

        # first element in first row MUST NOT be 0
        first_element = edited_matrix[0][0]
        assert (first_element != 0)

        # making first column filled with 0 (except first row)
        for i in range(1, size):
            coefficient = edited_matrix[i][0] / first_element
            for j in range(0, size):
                edited_matrix[i][j] -= coefficient * edited_matrix[0][j]

        sliced_matrix = self.get_slice(row=1, column=1, matrix=edited_matrix)
        return edited_matrix[0][0] * self.determinant(matrix=sliced_matrix, sign=sign)

    def det(self):
        return self.determinant()

    def swap_columns(self, c1, c2, matrix=None):
        """ expecting values from '1' to 'size' of matrix """
        if matrix is None:
            matrix = self.matrix

        size = self.get_size(matrix=matrix)
        c1 -= 1; c2 -= 1 # indexing from '0' to 'size-1'
        assert (0 <= c1 < size) & (0 <= c2 < size) # check if these columns exists

        column1 = tuple([row[c1] for row in matrix])
        column2 = tuple([row[c2] for row in matrix])

        for i in range(size):
            # exchanging values in each row
            matrix[i][c1] = column2[i]
            matrix[i][c2] = column1[i]

        del(column1, column2)
        return True

    def swap_rows(self, r1, r2, matrix=None):
        """ expecting values from '1' to 'size' of matrix """
        if matrix is None:
            matrix = self.matrix

        size = self.get_size(matrix=matrix)
        r1 -= 1; r2 -= 1 # indexing from '0' to 'size-1'
        assert (0 <= r1 < size) & (0 <= r2 < size) # check if these rows exists

        backup_row = matrix[r1][:]

        # exchanging values in each row
        matrix[r1] = matrix[r2]
        matrix[r2] = backup_row

        del(backup_row)
        return True

    def get_size(self, matrix=None):
        """ returning number of rows as a size of square matrix
        """
        if matrix is None:
            matrix = self.matrix

        return len(matrix)

    def copy(self, matrix=None):
        """ returning copy of given matrix
        """
        if matrix is None:
            matrix = self.matrix

        new_matrix = []
        for row in matrix:
            new_matrix.append(row[:])

        return new_matrix

    def get_slice(self, row=1, column=1, matrix=None):
        """ slicing and returing new matrix (list of lists) composed from elements
            below the 'row' and to the right of the 'column'
            of the given matrix

            by default returns slicing by first row and first column
        """
        if matrix is None:
            matrix = self.matrix

        size = self.get_size(matrix=matrix)
        row -= 1; column -= 1 # indexing from '0' to 'size-1'
        assert (0 <= row < size) & (0 <= column < size) # check if these columns exists

        new_matrix = []
        for i in range(row+1, size):
            new_row = matrix[i][(column+1):]
            new_matrix.append(new_row)
            del(new_row)

        if new_matrix == []:
            new_matrix == [[]]
        return new_matrix

    def is_null_column(self, column, matrix=None):
        """ checks is given column consists of 0 """
        if matrix is None:
            matrix = self.matrix

        size = self.get_size(matrix=matrix)
        column -= 1 # indexing from '0' to 'size-1'
        assert (0 <= column < size) # check if these columns exists

        for row in matrix:
            assert row[column:] # check for existence

            if row[column]:
                # not equal to 0 then
                return False

        return True

    def is_null_row(self, row, matrix=None):
        """ checks is given row consists of 0 """
        if matrix is None:
            matrix = self.matrix

        size = self.get_size(matrix=matrix)
        row -= 1 # indexing from '0' to 'size-1'
        assert (0 <= row < size) # check if these columns exists

        assert matrix[row:] # check for existence

        not_only_zeroes_in_row = True in [bool(value) for value in matrix[row]]
        return (not_only_zeroes_in_row == False)


if __name__ == '__main__':

    ##########################
    ### Manual input:
    # n = int(input())
    # A = SquareMatrix(n)

    # list_of_lists = []
    # for each in range(n):
        # list_of_lists.append([int(value) for value in input().split()])

    # A.fill_from(list_of_lists)
    # print('A.determinant:', A.determinant())

    ##########################
    ### Custom testing:
    # A = SquareMatrix(3)
    # list_of_lists = [
        # [0, 5, -4],
        # [3, 2,  1],
        # [2, -4, 5]
    # ]
    # A.fill_from(list_of_lists)

    # print('A.determinant:', round(A.determinant()))

    ##########################
    ### Main:
    A = SquareMatrix(5)
    print('A:')
    print(A)

    print('A.is_null_column(5)', A.is_null_column(5))
    print('A.is_null_row(4)', A.is_null_row(4))
    print('A.is_null_row(5)', A.is_null_row(5))

    list_of_lists = [
        [1,  2,     3,   4,     5,  0.6],
        [5,  4,   3.3, 2.2,  1.12],
        [0,  0, -4.55,   3,     4],
        [1,  2],
        [3]
    ]
    A.fill_from(list_of_lists)
    print(A)

    print('A.is_null_column(5)', A.is_null_column(5))
    print('A.is_null_row(4)', A.is_null_row(4))
    print('A.is_null_row(5)', A.is_null_row(5))

    A.swap_columns(3, 5)
    A.swap_rows(1, 2)
    print(A)

    B = SquareMatrix(A.get_size()-1)
    B.fill_from(A.get_slice())
    print('B (slice from A):')
    print(B)

    print('A.determinant:', A.determinant())
