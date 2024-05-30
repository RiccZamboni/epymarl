from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers.non_shared_controller import NonSharedMAC
import torch as th


# create ExpShareMAC class that inherits from NonSharedMAC
class ExpShareMAC(NonSharedMAC):

    def forward(self, ep_batch, t, test_mode=False, agent_id='dummy'):
        # print('agent_id:', agent_id)
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        # modify agent forward function to take agent_id as input
        # this function should return the action output such that only agent_id agent rollouts using ...
        # ...batch of all agents (instead each agent_id corresponding to each agent in the batch, as done in NonSharedMAC)
        # e.g. agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, agent_id)
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)