import timeit

def main():
    perform_cpu_load()


def perform_cpu_load():
    print("AM: Start...")
    my_time = timeit.timeit('"-".join(str(i) for i in range(99999))', number=1000)
    my_time_int = int(round(my_time, 0))

    print(my_time_int)


if __name__ == "__main__": main()