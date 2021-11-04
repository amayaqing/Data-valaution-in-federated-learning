BATCH_SIZE = 100
NUM_AGENT = 5           # number of participants
NUM_ROUND = 20          # number of training rounds
NUM_LOCAL_DATA = 1000   # number of local data

NOISE_ADD = False       # add noise
UNBALANCE = False       # assign different size to clients
NON_IID = False         # non-iid environment

NOISE_ADD_LATER = False     # add noise in the intermediate round
UNBALANCED_LATER = False    # assign different size in the intermediate round

learning_rate = 0.1
dv_method ='RDV'       # RDV, ksub, tmc, cluster, def
cluster_num = 3            # for cluter-based approach