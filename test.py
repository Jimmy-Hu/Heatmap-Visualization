import time
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    linked_list = []
    np_time_stamp = []
    list_time_stamp = []
    array1 = np.zeros((1, 1000, 1000), dtype = np.int8)
    array2 = np.ones((1, 1000, 1000), dtype = np.int8)

    start = time.time()
    np_data = np.empty((0, 1000, 1000), dtype = np.int8)
    for _ in range(100):
        np_data = np.append(np_data, array1 + array2, axis = 0)
        np_time_stamp.append(time.time() - start)

    start = time.time()
    list_data = []
    for _ in range(100):
        list_data.append(array1 + array2)
        list_time_stamp.append(time.time() - start)

    print(np_time_stamp)
    print(list_time_stamp)

    plt.figure(figsize = (10, 10))  
    plt.plot(np_time_stamp)
    plt.title("Numpy concat spend time")

    plt.figure(figsize = (10, 10))  
    plt.plot(list_time_stamp)
    plt.title("Linked list append spend time")
    plt.show()
        