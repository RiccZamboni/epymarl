# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
import numpy as np
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from controllers import REGISTRY as mac_REGISTRY
from components.standarize_stream import RunningMeanStd
from learners.ppo_learner import PPOLearner
from sacred.observers import FileStorageObserver
from scipy.spatial import distance


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
        self.lambda_update_t = -self.args.lambda_update_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        if self.args.save_policy_update_distance_matrix:
            self.policy_dist_update_t = -self.args.policy_distance_interval - 1
            self.saved_macs = []

        # set selective lambda matrix before the training
        if self.args.lambda_matrix=='selective':
            self.selective_lambda_matrix = th.eye(self.n_agents).to(self.args.device)


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

        self.policy_distance_array = th.zeros(self.args.epochs, self.n_agents, self.n_agents)

        for k in range(self.args.epochs):

            actor_logs = {
                    'clipped_loss': [],
                    'entropy_loss': [],
                    'is_ratio_mean': [],
            }

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

            # compute js among all agents 
            with th.no_grad():
                if self.args.metric == 'js':
                    # Jensen-Shannon distance
                    policy_distance_matrix = self.compute_js_matrix(pi)
                else:
                    raise NotImplementedError('only js metric is supported')
                self.policy_distance_array[k] = policy_distance_matrix

            if self.args.save_policy_update_distance_matrix:
                if k==0:
                    self.update_policy_distance_matrix(t_env, self.mac, batch)


            # define lambda as a vector of ones of n_agents size (later, will be given as the parameter in the config)
            if self.args.lambda_matrix=='one':
                lambda_matrix = th.ones((self.n_agents, self.n_agents), device=batch.device)
            elif self.args.lambda_matrix=='diag':
                lambda_matrix = th.eye(self.n_agents, device=batch.device)
            elif self.args.lambda_matrix=='selective':
                lambda_matrix = self.selective_lambda_matrix
            else:
                raise NotImplementedError('lambda_matrix should be one or diag')
            lambda_matrix = lambda_matrix.reshape(1, 1, self.n_agents, self.n_agents).repeat(batch.batch_size, batch.max_seq_length-1, 1, 1)

            advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards, critic_mask, ratios, lambda_matrix)
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
            entropy_loss = -(self.args.entropy_coef * entropy  * lambda_matrix * mask).sum() / mask.sum()
            seppo_loss = clipped_loss + entropy_loss

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

    def update_selective_lambda_matrix(self, t_env):

        if (t_env - self.lambda_update_t >= self.args.lambda_update_interval) and (t_env > self.args.lambda_update_start):

            self.selective_lambda_matrix = th.eye(self.n_agents).to(self.args.device)

            if self.args.lambda_select_method == 'cutoff':
                kl_matrix = self.policy_distance_array[0, :, :]
                delta = self.args.lambda_cutoff
                self.selective_lambda_matrix[ kl_matrix < delta ] = 1.0

            self.lambda_update_t = t_env

    def update_policy_distance_matrix(self, t_env, mac, batch):
        if (t_env - self.policy_dist_update_t >= self.args.policy_distance_interval):
            print('updating policy distance matrix at t_env:', t_env)
            with th.no_grad():
                new_mac = mac_REGISTRY[self.args.mac]( batch.scheme, batch.groups , self.args)
                new_mac.load_state(mac)
                self.saved_macs.append(new_mac)
                probs_list = self.get_probs_list(batch)

                if self.args.metric == 'js':
                    distance_fn = self.compute_js_distance
                else:
                    raise NotImplementedError('only js metric is supported')

                # generate a policy update distance matrix of size n_agents*len(log_ratios_list) x n_agents*len(log_ratios_list)
                policy_update_distance_matrix = th.zeros(len(probs_list)*self.n_agents, len(probs_list)*self.n_agents)
                for i,probs_i in enumerate(probs_list):
                    for j,probs_j in enumerate(probs_list):
                        for agent_id_i in range(self.n_agents):
                            for agent_id_j in range(self.n_agents):
                                policy_update_distance_matrix[i*self.n_agents+agent_id_i, j*self.n_agents+agent_id_j] =  distance_fn(probs_i[:,:,:,agent_id_i], probs_j[:,:,:,agent_id_j])

                # save the policy update distance matrix as csv in sacred directory
                self.save_to_csv(policy_update_distance_matrix.numpy(), t_env)

            self.policy_dist_update_t = t_env

    def save_to_csv(self, policy_update_distance_matrix, t_env):
        # get the path to save the policy update distance matrix
        run_obj = self.logger._run_obj
        file_observer = next(observer for observer in run_obj.observers if isinstance(observer, FileStorageObserver))
        file_path = file_observer.dir

        # save the policy update distance matrix as csv
        file_name = 'policy_update_distance_matrix.csv'
        file_path = file_path + '/' + file_name
        np.savetxt(file_path, policy_update_distance_matrix, delimiter=",")

        # add footer with t_env, n_agents, n_policies
        with open(file_path, 'a') as f:
            f.write('t_env: '+str(t_env)+'\n')
            f.write('n_agents: '+str(self.n_agents)+'\n')
            f.write('n_policies: '+str(len(self.saved_macs)))

    def get_probs_list(self, batch):
        probs_list = []
        for mac in self.saved_macs:
            probs_list.append(self.get_probs(mac, batch))
        return probs_list

    def get_probs(self, mac, batch):       

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        mask = mask.repeat(1, 1, self.n_agents)

        mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.n_agents)

        mac_out = []
        mac.init_hidden(batch.batch_size * self.n_agents)
        for t in range(batch.max_seq_length - 1):
            agent_outs = mac.forward( batch, t=t, all_agents=True )
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        pi = mac_out
        pi[mask == 0] = 1.0

        return pi

    def compute_js_distance(self, probs_i, probs_j):
        probs_i = probs_i.reshape(-1, probs_i.shape[-1] )
        probs_j = probs_j.reshape(-1, probs_j.shape[-1] )
        # compute js
        js = distance.jensenshannon( probs_i.transpose(0,1), probs_j.transpose(0,1), base=2).mean() 
        return th.tensor(js)
    
    def compute_js_matrix(self, probs):
        kl_matrix = th.zeros(self.n_agents, self.n_agents)
        for agent_id_i in range(self.n_agents):
            for agent_id_j in range( agent_id_i+1 , self.n_agents):
                kl_matrix[agent_id_i, agent_id_j] = self.compute_js_distance(probs[:,:,:,agent_id_i], probs[:,:,:,agent_id_j])
        kl_matrix = kl_matrix + kl_matrix.t() #symmetric matrix
        return kl_matrix
    

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
            seppo_critic_loss = ( (masked_td_error**2) * lambda_matrix *  clamped_ratios.detach() ).sum() / mask.sum()
        else:
            with th.no_grad():
                # log critic loss for each agent
                for agent_id in range(self.n_agents):
                    agent_loss = ( (masked_td_error[:,:,:,agent_id] ** 2 ) * lambda_matrix[:,:,agent_id,:]  ).sum() / mask[:,:,:,agent_id].sum()
                    # print(' agent_id:', agent_id, ' critic_loss:', agent_loss.item())
                    running_log["critic_loss"].append(agent_loss.item())
            seppo_critic_loss = ( (masked_td_error**2) * lambda_matrix  ).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        seppo_critic_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        return masked_td_error, running_log
