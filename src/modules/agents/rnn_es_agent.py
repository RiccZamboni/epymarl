from modules.agents.rnn_ns_agent import RNNNSAgent
import torch as th

# create RNNESAgent class that inherits from RNNSAgent

class RNNESAgent(RNNNSAgent):
    # define a modified forward function that accepts agent_id as an argument
    # this function should return the action output such that only agent_id agent rollouts using ...
    # ...batch of all agents (instead each agent_id corresponding to each agent in the batch, as done in NonSharedMAC)

    def forward(self, inputs, hidden_state, agent_id=None):

        if agent_id is None:
            # implement forward function from RNNNSAgent
            return super().forward(inputs, hidden_state)
        
        hiddens = []
        qs = []
        if inputs.size(0) == self.n_agents:
            for i in range(self.n_agents):
                q, h = self.agents[agent_id](inputs[i].unsqueeze(0), hidden_state[:, i])
                hiddens.append(h)
                qs.append(q)
            return th.cat(qs), th.cat(hiddens).unsqueeze(0)
        else:
            for i in range(self.n_agents):
                inputs = inputs.view(-1, self.n_agents, self.input_shape)
                q, h = self.agents[agent_id](inputs[:, i], hidden_state[:, i])
                hiddens.append(h.unsqueeze(1))
                qs.append(q.unsqueeze(1))
            return th.cat(qs, dim=-1).view(-1, q.size(-1)), th.cat(hiddens, dim=1)
        
