# Data valuation for Machine Learning and Federated Learning

This project designs a real-time round data valuation (RDV) scheme based on Shapley value to serve as an incentive scheme for Federated learning. The corresponding estimations are proposed to improve efficiency. At the same time, the data valuation idea is extended to optimize the federated aggregation process.

## Source code construction

The source code is composed by three major parts:

- **Main running part**:

  - *main.py*: the main running program for data valuation (incentive calculation) part. The output records all round Shapley value assigned to each participant. 
  - *FedOpt_dv.py*: the Positive-Only and Positive-Weighted strategies to optimize the model aggregation process (only include the baseline average aggregation method). The program performs the corresponding methods to obtain the global model and returns the accuracy value for each round.
  - *FedOpt_ransac.py*: Similar as *FedOpt_dv*, this program utilized RANSAC-selective approach to perform federated aggregation and return the accuracy value for each round.

- **Utility functions and parameter part**:

  - *glob.py*: defines and sets all global parameters needed for running the main program, including the number of local clients, the number of local data and what data environment is created, etc. 
  - *util.py* (in *Util/* folder): defines all necessary functions, like the `load_data`, `init_model`, etc. The implementation of all federated learning related operations, such as `local_train`, `federated_train`,  are borrowed from the official methods given by Google (https://www.tensorflow.org/federated). 
  - *plot.py* (in *Util/* folder): plot the data valuation results and store the graphs in the corresponding folders (Image/...).

- **Round-based data valuation implementation part**:

  This part is located in *DV/* folder, including the implementation of RDV, K-subset DV, TMC-DV, clusterDV and the orignial DefDV.



## Environment

Python 3.7

TensorFlow 2.3.1

TensorFlow Federated 0.17.0

```shell
pip install --upgrade tensorflow_federated
```



## Running

Change parameters in  *glob.py*, then 

```shell
python main.py           # for data valuation
python FedOpt_dv.py      # using data valuation to optimize aggregation
python FedOpt_ransac.py  # using RANSAC-selective to optimize aggregation
```



