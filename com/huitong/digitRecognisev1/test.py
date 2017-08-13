#encoding=utf-8
import numpy as np

def fun1():
    a = np.zeros([10], dtype=np.float32)
    # a[0,0] = 1
    a[2] = 2
    a[5] = 6

    print a
    b = np.nonzero(a)
    print b[0]

def fun2():
    a = 'a'
    b = 'c'
    print (b-a)

def fun3():
    print __file__




if __name__ == "__main__":
    fun3()