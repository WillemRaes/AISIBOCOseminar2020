import threading
from demo.Config import Config
import numpy as np
import time

class DataGenThread(threading.Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(DataGenThread, self).__init__()
        self.target = target
        self.name = name
        return

    def run(self):
        put_conn = False
        X_test = np.load("X_test.npy")

        for tcp_conn in X_test:
            put_conn = False
            while not put_conn:
                if not Config.q.full():
                    # print(tcp_conn.shape)
                    # print(tcp_conn)
                    Config.q.put(tcp_conn)
                    put_conn = True
                    # time.sleep(Config.deadTime_arr[Config.deadTime_sel])

                else:
                    time.sleep(0.005)


        return








