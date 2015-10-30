__author__ = 'Colo'

def add(a, b):
    """
    Add two numbers.
    :param a: operand a
    :param b: operand b
    :return: sum a + b

    >>> add(3, 5)
    8
    """
    pass

if __name__ == '__main__':
    import doctest
    doctest.testmod(verbose=True)
    print help(add)