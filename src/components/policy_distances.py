import numpy as np
import torch as th
from scipy.spatial import distance
from controllers import REGISTRY as mac_REGISTRY
from sacred.observers import FileStorageObserver

class PolicyDistances:
    '''
    This class is used to compute the distance between the policies of the agents.
    The distance is computed using the Jensen-Shannon divergence. In future, other
    distance metrics can be added.
    Currently, the class is tested with only IPPO and SEPPO algorithms.
    '''
    def __init__(self, logger, args):
        self.logger = logger
        self.args = args
        if self.args.save_policy_update_distance_matrix:
            self.policy_dist_update_t = -self.args.policy_distance_interval - 1
            self.saved_macs = []

    def compute_js_distance(self, probs_i, probs_j):

        # shape manupulation for compatibility with js function
        probs_i = probs_i.transpose(0,1)
        probs_j = probs_j.transpose(0,1)

        # compute js
        js = distance.jensenshannon( probs_i, probs_j, base=2).mean() 
        return th.tensor(js)
    
    def compute_js_matrix(self, probs):

        # shape manupulation for compatibility with all the learners
        probs = probs.reshape(-1, self.args.n_agents, self.args.n_actions)

        # compute js matrix
        js_matrix = th.zeros(self.args.n_agents, self.args.n_agents)
        for agent_id_i in range(self.args.n_agents):
            for agent_id_j in range( agent_id_i+1 , self.args.n_agents):
                js_matrix[agent_id_i, agent_id_j] = self.compute_js_distance(probs[:,agent_id_i], probs[:,agent_id_j])
        js_matrix = js_matrix + js_matrix.t() #symmetric matrix
        return js_matrix
    
    def update_all_pi_distance_matrix(self, t_env, mac, batch):
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
                policy_update_distance_matrix = th.zeros(len(probs_list)*self.args.n_agents, len(probs_list)*self.args.n_agents)
                for i,probs_i in enumerate(probs_list):
                    for j,probs_j in enumerate(probs_list):
                        for agent_id_i in range(self.args.n_agents):
                            for agent_id_j in range(self.args.n_agents):
                                policy_update_distance_matrix[i*self.args.n_agents+agent_id_i, j*self.args.n_agents+agent_id_j] =  distance_fn(probs_i[:,agent_id_i], probs_j[:,agent_id_j])

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
            f.write('n_agents: '+str(self.args.n_agents)+'\n')
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
        mask = mask.repeat(1, 1, self.args.n_agents)

        mac_out = []
        if self.args.name=="seppo":
            mac.init_hidden(batch.batch_size * self.args.n_agents)
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, self.args.n_agents)
        elif self.args.name=="ippo":
            mac.init_hidden(batch.batch_size)
        else:
            raise NotImplementedError('only supports ippo and seppo algorithms')

        for t in range(batch.max_seq_length - 1):
            if self.args.name=="seppo":
                agent_outs = mac.forward( batch, t=t, all_agents=True )
            elif self.args.name=="ippo":
                agent_outs = mac.forward( batch, t=t)
            else:
                raise NotImplementedError('only supports ippo and seppo algorithms')
            
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        pi = mac_out
        pi[mask == 0] = 1.0

        # for ippo, repeat pi for n_agents since same policy is shared among the agents
        if self.args.name == 'ippo':
            pi = pi.unsqueeze(-2).repeat(1, 1, 1, self.args.n_agents, 1)

        return pi.reshape(-1, self.args.n_agents, self.args.n_actions)
