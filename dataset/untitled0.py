from multiprocessing import Pool
import multiprocessing

def z(x):
    for i in range(1, 50):
        print(i)

p1 = multiprocessing.Process(target = z)
p2 = multiprocessing.Process(target = z)
