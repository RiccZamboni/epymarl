# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from components.standarize_stream import RunningMeanStd


class SEPPOLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)


        mask = mask.repeat(1, 1, self.n_agents)

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[mask == 0] = 1.0

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # repeat rewards, old_log_pi_taken, actions, mask to have shape (n_agents, nstep, eplength, n_agents)
        # to match data of all agents being fed through each agent's network in single batch
        rewards = rewards.unsqueeze(0).repeat(self.n_agents, 1, 1, 1)
        old_log_pi_taken = old_log_pi_taken.unsqueeze(0).repeat(self.n_agents, 1, 1, 1)
        actions = actions.unsqueeze(0).repeat(self.n_agents, 1, 1, 1, 1)
        mask = mask.unsqueeze(0).repeat(self.n_agents, 1, 1, 1)

        critic_mask = mask.clone()

        # # add a boolean vector to continue training for each agent
        # continue_training = [True for _ in range(self.args.n_agents)]

        for k in range(self.args.epochs):

            # if not all(continue_training): # early stopping if kl_divergence is more than kl_target
            #     break

            actor_logs = {
                    'clipped_loss': [],
                    'entropy_loss': [],
                    'is_ratio_mean': [],
                    # 'agent_grad_norm': [],
                    # 'advantages_mean': [],
                    # 'pi_max': []
            }

            
            # if not continue_training[agent_id]: # don't train agents with kl-div higher than kl_target
            #     continue

            agent_id=-1

            mac_out = []
            self.mac.init_hidden(batch.batch_size * self.n_agents)
            for t in range(batch.max_seq_length - 1):
                # create forward loop in mac s.t it rollouts using given agent_id
                # this forward loop creates agents_outs using the batch of all agents but with only agent_id (instead each agent_id rolling out barch data from each agent seprately as default behaviour in not-shared macs)
                # e.g. agent_outs = self.mac.forward(batch, t=t, agent_id=agent_id)
                agent_outs = self.mac.forward( batch, t=t, all_agents=True )
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=2)  # Concat over time

            pi = mac_out
            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=-1, index=actions).squeeze(-1)
            log_pi_taken = th.log(pi_taken + 1e-10)

            log_ratios = log_pi_taken - old_log_pi_taken.detach()
            ratios = th.exp(log_ratios)

            # print('ratios', ratios.size())

            # # early stopping of training for agent_id if kl_divergence is too high 
            # # inspired from discussion on https://github.com/DLR-RM/stable-baselines3/issues/417
            # # derivation can be found in Schulman blog: http://joschu.net/blog/kl-approx.html
            # if self.args.kl_target is not None:
            #     # compute approximated kl divergence between old policies of all agents and new policy of agent_id
            #     with th.no_grad():
            #         approx_kl_div = ( (ratios - 1) - log_ratios ).mean(dim=(0,1)).cpu().numpy()
            #     if approx_kl_div.max() > 1.5*self.args.kl_target:
            #         self.logger.console_logger.info('Early stopping at epoch {} for agent id {}'.format(k+1, agent_id))
            #         continue_training[agent_id] = False

            # create critic sqeuential such that only agent_id is used for training
            # can be done by adding agent_id as an argument to the function
            # e.g. advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards, critic_mask, agent_id)
            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask, ratios)
            advantages = advantages.detach()
            # Calculate policy grad with mask

            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            # define lambda as a vector of ones of n_agents size
            lambda_vector = th.ones(self.args.n_agents)

            # redefine pg_loss as agent-wise multiplication of lambda and pg_loss
            clipped_loss = -((th.min(surr1, surr2)) * lambda_vector * mask).sum() / mask.sum()
            entropy_loss = -((self.args.entropy_coef * entropy) * lambda_vector * mask).sum() / mask.sum()
            pg_loss = clipped_loss + entropy_loss

            # The below part will be out of the for loops of agent_id
            # replace pg_loss with seppo_loss 

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # log pg_loss, entropy_loss, is_ratios_mean for each agent_id
        for agent_id in range(self.n_agents):

            clipped_loss_log = (surr1[agent_id] * lambda_vector[agent_id] * mask[agent_id]).sum() / mask[agent_id].sum()
            entropy_loss_log = (self.args.entropy_coef * entropy[agent_id] * lambda_vector[agent_id] * mask[agent_id]).sum() / mask[agent_id].sum()
            is_ratio_mean_log = ratios[agent_id].mean().item()

            actor_logs['clipped_loss'].append(clipped_loss_log.item())
            actor_logs['entropy_loss'].append(entropy_loss_log.item())
            actor_logs['is_ratio_mean'].append(is_ratio_mean_log)


        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # logging should be done for each agent_id
        with th.no_grad():
            if t_env - self.log_stats_t >= self.args.learner_log_interval:

                # critc logging
                for agent_id in range(self.n_agents):               
                    for key in critic_train_stats:
                        self.logger.log_stat('agent_'+str(agent_id)+'/'+key, critic_train_stats[key][agent_id], t_env)

                # actor logging
                for agent_id in range(self.n_agents):
                    for key in actor_logs:
                        self.logger.log_stat('agent_'+str(agent_id)+'/'+key, actor_logs[key][agent_id], t_env)

            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, ratios):
        # accept agent_id as an argument

        # check if we can keep this critic forward our of the for loop since the batch stays the same for all epochs 
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(-1)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
            # "critic_grad_norm": [],
            # "td_error_abs": [],
            # "target_mean": [],
            # "q_taken_mean": [],
        }

        # check if we really need the importance sampling for the critic (My guess is that we don't need it since there is no assumption that critic network expects data from the same distribution)

        # critic forward function needs to be modified to accept agent_id as an argument
        # This new forward function will only return the critic values with the critic for given agent_id
        v = critic(batch)[:, :, :-1].squeeze(-1)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        if self.args.use_critic_importance_sampling:
            # log critic loss for each agent
            with th.no_grad():
                for agent_id in range(self.n_agents):
                    agent_loss = ( (masked_td_error[agent_id]**2) * ratios[agent_id].detach() ).sum() / mask[agent_id].sum()
                    running_log["critic_loss"].append(agent_loss.item())
            loss = ( (masked_td_error**2) * ratios.detach() ).sum() / mask.sum()
        else:
            # log critic loss for each agent
            with th.no_grad():
                for agent_id in range(self.n_agents):
                    agent_loss = (masked_td_error[agent_id] ** 2).sum() / mask[agent_id].sum()
                    running_log["critic_loss"].append(agent_loss.item())
            loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        # running_log["critic_loss"].append(loss.item())
        # running_log["critic_grad_norm"].append(grad_norm.item())
        # mask_elems = mask.sum().item()
        # running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
        # running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        # running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        ep_length = rewards.size(2)
        nstep_values = th.zeros_like(values[:, :, :-1])
        for t_start in range(ep_length):
            nstep_return_t = th.zeros_like(values[:, :, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= ep_length:
                    break
                elif step == nsteps:
                    nstep_return_t += (
                        self.args.gamma**step * values[:, :, t] * mask[:, :, t]
                    )
                elif t == ep_length - 1 and self.args.add_value_last_step:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, :, t] * mask[:, :, t]
                    )
                    nstep_return_t += (
                        self.args.gamma ** (step + 1) * values[:, :, t + 1]
                    )
                else:
                    nstep_return_t += (
                        self.args.gamma**step * rewards[:, :, t] * mask[:, :, t]
                    )
            nstep_values[:, :, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
