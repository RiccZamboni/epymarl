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
from learners.ppo_learner import PPOLearner


class SEPPOLearner(PPOLearner):
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

        # reshape/ repeat from (batch_size, eplength, n_agents) to (batch_size, eplength, n_agents, n_agents)
        # to match data of all agents being fed through each agent's network in single batch with
        # entry [:, :, i, :] being the data of agent i
        rewards = rewards.unsqueeze(-1).repeat(1, 1, 1, self.n_agents)
        old_log_pi_taken = old_log_pi_taken.unsqueeze(-1).repeat(1, 1, 1, self.n_agents)
        actions = actions.unsqueeze(-2).repeat(1, 1, 1, self.n_agents, 1)
        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.n_agents)

        critic_mask = mask.clone()

        for k in range(self.args.epochs):

            actor_logs = {
                    'clipped_loss': [],
                    'entropy_loss': [],
                    'is_ratio_mean': [],
            }

            # set logs for kl_div among agents
            if self.args.log_kl:
                for agent_id in range( self.n_agents):
                    actor_logs['kl_with_agent_'+str(agent_id)+'_epoch_'+str(k)] = []

            mac_out = []
            self.mac.init_hidden(batch.batch_size * self.n_agents)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward( batch, t=t, all_agents=True )
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            pi[mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=-1, index=actions).squeeze(-1)
            log_pi_taken = th.log(pi_taken + 1e-10)

            log_ratios = log_pi_taken - old_log_pi_taken.detach()
            ratios = th.exp(log_ratios)

            # kl logging: approximated kl divergence among all agents
            # derivation can be found in Schulman blog: http://joschu.net/blog/kl-approx.html
            if self.args.log_kl:
                # compute approximated kl divergence among all agents 
                with th.no_grad():
                    kl_matrix = th.zeros(self.n_agents, self.n_agents)
                    for agent_id_i in range(self.n_agents):
                        for agent_id_j in range(self.n_agents):
                            log_ratios_ij = log_ratios[:,:,:,agent_id_i] - log_ratios[:,:,:,agent_id_j]
                            ratios_ij = th.exp(log_ratios_ij)
                            kl_matrix[agent_id_i, agent_id_j] = ( (ratios_ij - 1) - log_ratios_ij ).mean()

                        # log kl_divergence for each agent_id_i
                        for kl in kl_matrix[agent_id_i]:
                            actor_logs['kl_with_agent_'+str(agent_id_i)+'_epoch_'+str(k)].append(kl.item())

            # define lambda as a vector of ones of n_agents size (later, will be given as the parameter in the config)
            if self.args.lambda_matrix=='one':
                lambda_matrix = th.ones((self.n_agents, self.n_agents), device=batch.device)
            elif self.args.lambda_matrix=='diag':
                lambda_matrix = th.eye(self.n_agents, device=batch.device)
            else:
                raise NotImplementedError('lambda_matrix should be one or diag')
            lambda_matrix = lambda_matrix.reshape(1, 1, self.n_agents, self.n_agents).repeat(batch.batch_size, batch.max_seq_length-1, 1, 1)

            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                      critic_mask, ratios, lambda_matrix)
            advantages = advantages.detach()
            # Calculate policy grad with mask

            surr1 = ratios * advantages
            surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

            # print('t_env:', t_env, ' epoch:', k)
            for agent_id in range(self.n_agents):
                with th.no_grad():
                    clipped_loss = -(  (th.min(surr1[:,:,:,agent_id], surr2[:,:,:,agent_id])) * lambda_matrix[:,:,agent_id,:]  * mask[:,:,:,agent_id]).sum() / mask[:,:,:,agent_id].sum()
                    entropy_loss = -(self.args.entropy_coef * entropy[:,:,:,agent_id]  * lambda_matrix[:,:,agent_id,:]  * mask[:,:,:,agent_id]).sum() / mask[:,:,:,agent_id].sum()
                    ratios_mean = ratios[:,:,:,agent_id].mean().item()

                    # print('agent:', agent_id, ' clipped loss:', clipped_loss.item(), 'entropy loss:', entropy_loss.item())
                    actor_logs['clipped_loss'].append(clipped_loss.item())
                    actor_logs['entropy_loss'].append(entropy_loss.item())
                    actor_logs['is_ratio_mean'].append(ratios_mean)

            clipped_loss = -(  (th.min(surr1, surr2)) * lambda_matrix  * mask).sum() / mask.sum()
            entropy_loss = -(self.args.entropy_coef * entropy  * lambda_matrix  * mask).sum() / mask.sum()
            seppo_loss = clipped_loss + entropy_loss

            # Optimise agents
            self.agent_optimiser.zero_grad()
            seppo_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()


        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # logging done for each agent_id
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

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, ratios, lambda_matrix):

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


        v = critic(batch)[:, :-1].squeeze(-1)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask

        if self.args.use_critic_importance_sampling:
            with th.no_grad():
                # log critic loss for each agent
                for agent_id in range(self.n_agents):
                    # clamp the importance sampling weights
                    clamped_ratios = th.clamp(ratios[:,:,:,agent_id], 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                    agent_loss = ( (masked_td_error[:,:,:,agent_id]**2) * lambda_matrix[:,:,agent_id,:]  *  clamped_ratios ).sum() / mask[:,:,:,agent_id].sum()
                    # print(' agent_id:', agent_id, ' critic_loss:', agent_loss.item())
                    running_log["critic_loss"].append(agent_loss.item())
            clamped_ratios = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
            seppo_critic_loss = ( (masked_td_error**2) * lambda_matrix  *  clamped_ratios.detach() ).sum() / mask.sum()
        else:
            with th.no_grad():
                # log critic loss for each agent
                for agent_id in range(self.n_agents):
                    agent_loss = ( (masked_td_error[:,:,:,agent_id] ** 2 ) * lambda_matrix[:,:,agent_id,:]  ).sum() / mask[:,:,:,agent_id].sum()
                    # print(' agent_id:', agent_id, ' critic_loss:', agent_loss.item())
                    running_log["critic_loss"].append(agent_loss.item())
            seppo_critic_loss = ( (masked_td_error**2) * lambda_matrix ).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        seppo_critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        return masked_td_error, running_log
