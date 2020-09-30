import queue
class Config:


    deadTime_arr = [0.005, 0.01, 0.1, 0.15, 0.2]  # dynamic delays
    deadTime_sel = 0  # position in deadTime_sel as determined in Consumerthread2
    bufsize = 32
    q = queue.Queue(maxsize=20)