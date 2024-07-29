# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.linalg import inv, det, sqrtm
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
        self.lambda_update_t = -self.args.lambda_update_interval - 1

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

        critic_mask = mask.clone()

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

        # critic stats dict to store each agent's critic stats
        critic_train_stats_all = {}

        # set logs for kl_dive among agents
        if self.args.kl_logs:
            kl_logs = {}
            for k in range(self.args.epochs):
                for agent_id in range( self.args.n_agents):
                    kl_logs['kl_with_agent_'+str(agent_id)+'_epoch_'+str(k)] = []

        if self.args.lambda_matrix=='selective':
            self.kl_array = th.zeros(self.args.epochs, self.n_agents, self.n_agents)

        # set selective lambda matrix at the beginning of training
        if self.args.lambda_matrix=='selective':
            self.selective_lambda_matrix = th.eye(self.n_agents).to(self.args.device)

        for k in range(self.args.epochs):

            seppo_loss=0

            # set logs
            actor_logs = {
                'clipped_loss': [],
                'entropy_loss': [],
                'is_ratio_mean': [],
            }

            # add for loop loop over all agents
            for agent_id in range( self.args.n_agents):

                mac_out = []
                self.mac.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length - 1):
                    # experience sharing controller
                    # rollouts only for given agent_id
                    agent_outs = self.mac.forward(batch, t=t, agent_id=agent_id )
                    mac_out.append(agent_outs)
                mac_out = th.stack(mac_out, dim=1)  # Concat over time

                pi = mac_out
                pi[mask == 0] = 1.0

                pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
                log_pi_taken = th.log(pi_taken + 1e-10)

                log_ratios = log_pi_taken - old_log_pi_taken.detach()
                ratios = th.exp(log_ratios)

                # approx kl_divergence calculation
                # inspired from discussion on https://github.com/DLR-RM/stable-baselines3/issues/417
                # derivation can be found in Schulman blog: http://joschu.net/blog/kl-approx.html
                if self.args.lambda_matrix=='selective':
                    # compute approximated kl divergence between old policies of all agents and new policy of agent_id
                    with th.no_grad():
                        approx_kl_div = ( (ratios - 1) - log_ratios ).mean(dim=(0,1)).detach()
                        self.kl_array[k, agent_id] = approx_kl_div

                # define lambda vector
                if self.args.lambda_matrix=='one':
                    lambda_vector = th.ones(self.args.n_agents)
                elif self.args.lambda_matrix=='diag':
                    lambda_vector = th.eye(self.args.n_agents)[agent_id]
                elif self.args.lambda_matrix=='selective':
                    lambda_vector = self.selective_lambda_matrix[agent_id]
                else:
                    raise NotImplementedError('Lambda matrix type not implemented; only one and diag are supported')
                lambda_vector = lambda_vector.view(1,1,-1).repeat(batch.batch_size, batch.max_seq_length-1, 1).to(self.args.device)

                advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
                                                                          critic_mask, ratios, agent_id, lambda_vector)
                critic_train_stats_all[agent_id] = critic_train_stats
                advantages = advantages.detach()

                surr1 = ratios * advantages
                surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

                entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)

                # redefine pg_loss as agent-wise multiplication of lambda and pg_loss
                clipped_loss = -((th.min(surr1, surr2)) * lambda_vector * mask).sum() / mask.sum()
                entropy_loss = -((self.args.entropy_coef * entropy) * lambda_vector * mask).sum() / mask.sum()
                pg_loss = clipped_loss + entropy_loss

                # add pg_loss to seppo_loss
                seppo_loss  = seppo_loss + pg_loss

                # update actor logs
                actor_logs['clipped_loss'].append(clipped_loss.item())
                actor_logs['entropy_loss'].append(entropy_loss.item())
                actor_logs['is_ratio_mean'].append(ratios.mean().item())

                # update kl logs
                if self.args.kl_logs:
                    for i,kl in enumerate(approx_kl_div):
                        kl_logs['kl_with_agent_'+str(agent_id)+'_epoch_'+str(k)].append(kl.item())

            # Optimise agents
            self.agent_optimiser.zero_grad()
            seppo_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
            self.agent_optimiser.step()

        # update selective lambda matrix
        if self.args.lambda_matrix=='selective':
                self.update_selective_lambda_matrix(t_env)

        self.old_mac.load_state(self.mac)

        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        # logging should be done for each agent_id
        if t_env - self.log_stats_t >= self.args.learner_log_interval:

            # critc logging
            for agent_id in range(self.n_agents):               
                critic_train_stats = critic_train_stats_all[agent_id]
                ts_logged = len(critic_train_stats["critic_loss"])
                for key in critic_train_stats:
                    self.logger.log_stat('agent_'+str(agent_id)+'/'+key, sum(critic_train_stats[key]) / ts_logged, t_env)

            # actor logging
            for agent_id in range(self.n_agents):
                for key in actor_logs:
                    self.logger.log_stat('agent_'+str(agent_id)+'/'+key, actor_logs[key][agent_id], t_env)

            # kl_divergence logging
            if self.args.kl_logs:
                for agent_id in range(self.n_agents):
                    for key in kl_logs:
                        self.logger.log_stat('agent_'+str(agent_id)+'/'+key, kl_logs[key][agent_id], t_env)

            self.log_stats_t = t_env

    def update_selective_lambda_matrix(self, t_env):

        if t_env - self.lambda_update_t >= self.args.lambda_update_interval:

            # get all kl values
            self.kl_array = self.kl_array.detach().cpu().numpy()
            # set diagoanl values to nan
            for agent_id in range(self.n_agents):
                self.kl_array[:, agent_id, agent_id] = np.nan

            # GMM analysis of all_kl_values
            # Fit a Gaussian Mixture Model with 2 components
            gmm = BayesianGaussianMixture(n_components=2)
            gmm.fit( self.kl_array[ ~np.isnan(self.kl_array) ].reshape(-1,1) )
            means = gmm.means_.flatten()
            covariances = gmm.covariances_.flatten()

            # update lambda matrix using Bhattacharyya coefficient of GMM components
            bhat_coef = self.bhattacharyya(means, covariances)
            if bhat_coef < 0.5:
                dist_index = np.argmin(means)
                kl_limit = means[dist_index] + np.sqrt(covariances[dist_index])
                kl_div = self.kl_array[-1, :, :]
                kl_div[ np.eye(self.n_agents, dtype=bool) ] = 0
                self.selective_lambda_matrix[ th.from_numpy(kl_div < kl_limit) ] = 1.0

            self.lambda_update_t = t_env

    def bhattacharyya(self, means, vars):

        """
        Calculate the Bhattacharyya coefficient between two Gaussian distributions
        """

        # Extract the means and covariances
        mean1 = means[0]
        mean2 = means[1]
        var1 = vars[0]
        var2 = vars[1]

        # Calculate the Bhattacharyya distance
        BC = 1/4 * ( (mean1 - mean2)**2 / (var1 + var2 + 1e-6) ) + 1/4 * np.log( (1/4) * (  var1/var2 + var2/var1 + 2 ) ) 

        # Calculate the Bhattacharyya coefficient
        B = np.exp(-BC)

        return B



    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, ratios, agent_id, lambda_vector):
        
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch, agent_id=agent_id)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        running_log = {
            "critic_loss": [],
        }

        # exp. sharing critic forward function rolls out using given agent_id 
        v = critic(batch, agent_id=agent_id)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        if self.args.use_critic_importance_sampling:
            loss = ( (masked_td_error**2) * lambda_vector * ratios.detach() ).sum() / mask.sum()
        else:
            loss = ((masked_td_error**2) * lambda_vector).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())

        return masked_td_error, running_log

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
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
