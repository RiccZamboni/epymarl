from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
from controllers.non_shared_controller import NonSharedMAC
import torch as th


# create ExpShareMAC class that inherits from NonSharedMAC
class ExpShareMAC(NonSharedMAC):

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, agent_id=None):
        if agent_id is None:
            # execute select_actions function from NonSharedMAC
            chosen_actions = super().select_actions(ep_batch, t_ep, t_env, bs, test_mode)
            return chosen_actions
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, agent_id=agent_id )
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions


    def forward(self, ep_batch, t, test_mode=False, all_agents=False):

        if not all_agents:
            # execute forward function from NonSharedMAC
            return super().forward(ep_batch, t, test_mode)

        batch_size = ep_batch.batch_size * self.n_agents
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]

        # repeat data for all agents in the batch
        agent_inputs = (
            agent_inputs.unsqueeze(0)
            .repeat(self.n_agents, 1, 1)
            .reshape(batch_size * self.n_agents, -1)
        )

        avail_actions = avail_actions.repeat(self.n_agents, 1, 1, 1).reshape(
            self.n_agents, batch_size, -1
        )
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # print('agent_outs[:50]', agent_outs[:50])
        # print('agent_outs[50:100]', agent_outs[50:100])
        # print('agent_outs[100:150]', agent_outs[100:150])
        # print('agent_outs[150:200]', agent_outs[150:200])
        # print('agent_outs[200:250]', agent_outs[200:250])

        # modify agent forward function to take agent_id as input
        # this function should return the action output such that only agent_id agent rollouts using ...
        # ...batch of all agents (instead each agent_id corresponding to each agent in the batch, as done in NonSharedMAC)
        # e.g. agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, agent_id)
        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, agent_id=agent_id)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
        return agent_outs.view( self.n_agents, ep_batch.batch_size, self.n_agents, -1 )
