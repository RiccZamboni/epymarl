# code adapted from https://github.com/wendelinboehmer/dcg

import torch
import torch.nn as nn
import torch.nn.functional as F



class ModularRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ModularRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.n_actions)

        # make similar network for combined reward
        self.fc1_comb = nn.Linear(input_shape, args.hidden_dim)
        if self.args.use_rnn:
            self.rnn_comb = nn.GRUCell(args.hidden_dim, args.hidden_dim)
        else:
            self.rnn_comb = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc2_comb = nn.Linear(args.hidden_dim, args.n_actions)


    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hidden_dim).zero_()

    def forward_ind_network(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h
    
    def forward_combined_network(self, inputs, hidden_state):
        x = F.relu(self.fc1_comb(inputs))
        h_in = hidden_state.reshape(-1, self.args.hidden_dim)
        if self.args.use_rnn:
            h = self.rnn_comb(x, h_in)
        else:
            h = F.relu(self.rnn_comb(x))
        q = self.fc2_comb(h)
        return q, h
    
    def forward(self, inputs, hidden_state, agent_v_vector):
        # reshape inputs to size (n_agents, batch_size, input_shape)
        n_agents = self.args.n_agents
        inputs = inputs.view(n_agents, -1, inputs.shape[-1])
        hidden_state = hidden_state.view(n_agents, -1, hidden_state.shape[-1])
        # agent_v_vector indicates whether an agent uses individual or combined network
        # if agent_v_ vector is ['c', 'c', 'i', 'c', 'c'], then agent 0, 1, 3, 4 use combined network and agent 2 uses individual network
        q = []
        h = []
        for i in range(len(agent_v_vector)):
            if agent_v_vector[i] == 'i':
                q_i, h_i = self.forward_ind_network(inputs[i], hidden_state[i])
            else:
                q_i, h_i = self.forward_combined_network(inputs[i], hidden_state[i])
            q.append(q_i)
            h.append(h_i)
        # convert list to tensor
        q = torch.stack(q).reshape(-1, self.args.n_actions)
        h = torch.stack(h).reshape(-1, self.args.hidden_dim)
        return q, h

