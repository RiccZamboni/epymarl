import torch.nn as nn
from modules.agents.rnn_agent import RNNAgent
import torch as th

class RNNNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.input_shape = input_shape
        self.agents = th.nn.ModuleList([RNNAgent(input_shape, args) for _ in range(self.n_agents)])

    def init_hidden(self):
        # make hidden states on same device as model
        return th.cat([a.init_hidden() for a in self.agents])

    def forward(self, inputs, hidden_state):
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[i](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[i](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            out_dim = qs[0].size(-1)
            # print('q shape', qs[0].shape)
            # print('q size', q.shape)
            # for i,q in enumerate(qs):
            #     print(f"Agent {i} Q-values: {q}")
            return th.cat(qs, dim=1).view(-1, out_dim), th.cat(hiddens, dim=1)
        
    # create another forward loop for exprience sharing which takes agent_id as additional input
    # This function should return the action output such that only agent_id agent rollouts using ...
    # ...batch of all agents (instead each agent_id corresponding to each agent in the batch, as done in NonSharedMAC)
    # e.g. def forward(self, inputs, hidden_state, agent_id):

    def cuda(self, device="cuda:0"):
        for a in self.agents:
            a.cuda(device=device)
