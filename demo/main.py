from demo.DataGenerator import DataGenThread
from demo.OD_onnx_runtime import OnnxRuntimeConsumer
import logging
logging.basicConfig(level=logging.DEBUG, format='(%(asctime)s %(threadName)-9s) %(message)s', filename="log.txt")
import multiprocessing
def main():

    c = OnnxRuntimeConsumer(name='Consumer-IsolationForest-OD', modellName='"IsolationForest.onnx"')
    c.start()
    logging.debug("Started Consumerthread")
    p = DataGenThread(name='Producer')
    p.start()
    logging.debug("Started producerthread")


if __name__ == '__main__':
    main()


