REGISTRY = {}

from .rnn_agent import RNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_es_agent import RNNESAgent
from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_es"] = RNNESAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
