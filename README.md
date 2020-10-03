# AISIBOCOseminar2020
Deploy Machine Learning Models on Single Board Computer with ONNX 

This project contains:
- the necesarry steps to build ONNX-runtime from source on Raspberry Pi4 (described in MLdeployONNX.sh).
- A demo project where a scikit-learn model (IsolationForest) is used for outlier detection (on KDD cup 99 TCP dataset).
- Conversion of the model to ONNX format using skl2onnx.
- Realtime outlier detector for Raspberry Pi4 using dummy datagenerator (sampling from test set).
