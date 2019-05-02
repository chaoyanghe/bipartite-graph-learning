from memory_profiler import profile

# @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
@profile
def my_func():
    a = [1] * (10 ** 6)
    b = [2] * (2 * 10 ** 7)
    del b
    return a


if __name__ == '__main__':
    my_func()
