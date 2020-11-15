import time

def timing(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()

        print(f"time elapse: {end_time - start_time:.6f} seconds")
    return wrapper


@timing
def func(a, b, c):
    print(a, b)
    print("š", c)


func(1, 'asad', c='wtf??')

