import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP
from modules.critics.rnn import RNN
from modules.critics.ac_ns import ACCriticNS


class ACCriticES(ACCriticNS):

    # Add a new forward function that accepts agent_id as an argument
    # This new forward function will only return the critic values with the critic for given agent_id

    def forward(self, batch, t=None, agent_id=None):

        if agent_id is None:
            raise ValueError("agent_id must be provided for ACCriticES")
        
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[agent_id](inputs[:, :, i])
            qs.append(q.view(bs, max_t, 1, -1))
        q = th.cat(qs, dim=2)
        return q

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch["obs"][:, ts])

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # agent id
        input_shape += self.n_agents
        return input_shape
    

