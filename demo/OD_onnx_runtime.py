import onnxruntime as rt
import sklearn
import numpy as np
from sklearn.metrics import confusion_matrix
import threading
from Config import Config
import time
import multiprocessing


class OnnxRuntimeConsumer(threading.Thread):
    def __init__(self, group=None, target=None, name=None, modellName=None,
                 args=(), kwargs=None, verbose=None):
        super(OnnxRuntimeConsumer, self).__init__()
        self.target = target
        self.name = name
        self.modelName = modellName
        return

    def run(self):
        detected = 0
        sess_setup_start = time.time()
        sess_options = rt.SessionOptions()
        sess_options.intra_op_num_threads = multiprocessing.cpu_count()
        sess = rt.InferenceSession(self.modelName)
        print("Session setup time: ", time.time() - sess_setup_start)
        input_name = sess.get_inputs()[0].name
        print("input name", input_name)
        input_shape = sess.get_inputs()[0].shape
        print("input shape", input_shape)
        input_type = sess.get_inputs()[0].type
        print("input type", input_type)
        label_name = sess.get_outputs()[0].name
        print("output name", label_name)
        output_shape = sess.get_outputs()[0].shape
        print("output shape", output_shape)
        output_type = sess.get_outputs()[0].type
        print("output type", output_type)

        Y_test = np.load("Y_test.npy")
        input_name = sess.get_inputs()[0].name
        input_buffer = []
        while True:

            if not Config.q.empty():
                tcp_conn = Config.q.get()
                input_buffer.append(tcp_conn)
                # input = np.array(tcp_conn, dtype=np.float32)# .astype(np.float32)
                if len(input_buffer) == Config.bufsize:
                    start = time.time()
                    pred_onx = sess.run([label_name], {input_name: np.array(input_buffer, dtype=np.float32)})[0]
                    print("Inference time: ", time.time() - start)
                    input_buffer = []
                    # print(pred_onx)
                    Config.deadTime_sel1 = int(Config.q.qsize() / len(Config.deadTime_arr))
                    num_outliers = np.sum(pred_onx.ravel() == -1)
                    if num_outliers > 0:
                        print("[+] Detected outliers in this batch")
                        detected += num_outliers
                        print("Total number detected outliers: ", detected)

            else:
                time.sleep(0.005)

        return


