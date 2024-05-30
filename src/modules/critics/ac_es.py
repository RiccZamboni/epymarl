import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP
from modules.critics.rnn import RNN
from modules.critics.ac_ns import ACCriticNS


class ACCriticES(ACCriticNS):

    # Add a new forward function that accepts agent_id as an argument
    # This new forward function will only return the critic values with the critic for given agent_id
    # e.g. def forward(self, batch, t=None, agent_id):
    def forward(self, batch, t=None, agent_id=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            if self.args.use_critic_rnn:
                q = self.critics[i](inputs[:, :, i], max_t, bs)
            else:
                q = self.critics[i](inputs[:, :, i])
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q
    

