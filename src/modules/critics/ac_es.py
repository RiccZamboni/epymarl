import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.critics.mlp import MLP
from modules.critics.rnn import RNN
from modules.critics.ac_ns import ACCriticNS


class ACCriticES(ACCriticNS):

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        qs = []
        for i in range(self.n_agents):
            q = self.critics[i](inputs[:, :, :, i])
            qs.append(q.view(bs, max_t, self.n_agents, -1))
        return th.stack(qs, dim=-2)

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        inputs.append( batch["obs"][:, ts] )

        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))
        inputs = th.cat(inputs, dim=-1)

        # repeat inputs by agents to feed all agents' experience to all agents' critics
        # in a single batch
        # reshape from (batch_size, ep_length + 1, n_agents, obs_shape)
        # to (n_agents * batch_size, ep_length + 1, n_agents, obs_shape)
        inputs = inputs.unsqueeze(-2).repeat(1, 1, 1, self.n_agents, 1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        # agent id
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
    
    

