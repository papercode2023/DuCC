from torch.optim import RMSprop, Adam
from torch.optim import RMSprop, Adam
import numpy as np
from utils.supcontrast_new import SupConLoss

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents

        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer == "qtran_base":
            self.mixer = QTranBase(args)
        elif args.mixer == "qtran_alt":
            raise Exception("Not implemented here!")

        self.params += list(self.mixer.parameters())
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        self.representation_optim = Adam(params=self.mac.agent.representation.parameters(), lr=0.0001)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.supconLoss = SupConLoss(contrast_mode='all')
        self.repre_train_step = 0
        self.stable_coeff = 0.001

    def update_phi(self, t_env: int, episode_num: int, buffer: None, batch_size: 32, writer: None, last_success_rate: 0.0):
        # update representation
        buffer.update_prio()
        if last_success_rate > 0.01:
            self.stable_coeff = 0.1

        p_lst = [[] for i in range(self.n_agents)]
        idx_lst = [[] for i in range(self.n_agents)]
        target_phi_reg = copy.deepcopy(self.mac.agent.representation)
        start_loss, end_loss = None, None
        start_time = time.time()
        contrast_loss = []
        reg_loss = []

        for j in range(self.args.update_repre_times):
            # print('j: ', j)
            individual_loss = []
            for i in range(self.n_agents):
                # start_time1 = time.time()
                # print('i: ', i)

                anchor_obs, positives_obs, negatives_obs, selected_idx, reg_obs = buffer.repre_sample(i)
                anchor_obs = anchor_obs.to(device=self.device)
                positives_obs = positives_obs.to(device=self.device)
                negatives_obs = negatives_obs.to(device=self.device)

                anchor = self.mac.agent.representation(anchor_obs)
                positives = self.mac.agent.representation(positives_obs)
                negatives = self.mac.agent.representation(negatives_obs)

                anchor_mask = (1 - anchor_obs.sum(1).eq(0.).float())
                pos_mask = (1 - positives_obs.sum(2).eq(0.).float()).permute(1, 0)
                neg_mask = (1 - negatives_obs.sum(2).eq(0.).float()).permute(1, 0)

                supcontrast_loss, p = self.supconLoss(anchor, positives, negatives, anchor_mask, pos_mask, neg_mask)
                p_lst[i].append(p)
                idx_lst[i].append(selected_idx)

                # infoNce
                # info_nce_loss = []
                # for idx in range(positives.size(1)):
                #     info_nce_loss.append(info_nce(anchor, positives[:, idx, :], negatives, negative_mode='paired'))
                # info_nce_loss = th.stack(info_nce_loss)
                # info_nce_loss = info_nce_loss.mean()
                # individual_loss.append(info_nce_loss)

                if self.args.add_reg and (self.repre_train_step > self.args.stable_interval):
                    reg_obs = th.tensor(reg_obs, dtype=th.float32).to(self.device)
                    with th.no_grad():
                        reg_feature_old = target_phi_reg(reg_obs)
                    reg_feature_new = self.mac.agent.representation(reg_obs)
                    stable_loss = (reg_feature_new - reg_feature_old).pow(2).mean()
                    supcontrast_loss = supcontrast_loss + stable_loss * self.stable_coeff

                individual_loss.append(supcontrast_loss)


            avg_ind_loss = th.stack(individual_loss).mean()
            self.representation_optim.zero_grad()
            avg_ind_loss.backward()
            self.representation_optim.step()

            contrast_loss.append(avg_ind_loss.detach().cpu().numpy())

            if self.args.add_reg:
                pass

            if self.repre_train_step % 1 == 0:
                writer.add_scalar('contrast_loss', avg_ind_loss.detach().cpu().numpy(), self.repre_train_step)
            self.repre_train_step += 1

        print("update phi time", time.time() - start_time)
        # print('repre_train_step ', self.repre_train_step)

        # update p
        for i in range(self.n_agents):
            p_array = np.array(p_lst[i])
            idx_array = np.array(idx_lst[i])
            p_array = p_array.reshape(-1, 1)
            idx_array = idx_array.reshape(-1, idx_array.shape[2])
            p_array = th.from_numpy(p_array)
            idx_array = th.from_numpy(idx_array)
            buffer.update_p(i, p_array, idx_array)

        contrast_loss = np.mean(contrast_loss)

        return contrast_loss


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        obs = batch["obs"]

        # Calculate estimated Q-Values
        mac_out = []
        mac_hidden_states = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, _ = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            mac_hidden_states.append(self.mac.hidden_states)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        mac_hidden_states = th.stack(mac_hidden_states, dim=1)
        mac_hidden_states = mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        target_mac_hidden_states = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_hidden_states.append(self.target_mac.hidden_states)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[:], dim=1)  # Concat across time
        target_mac_hidden_states = th.stack(target_mac_hidden_states, dim=1)
        target_mac_hidden_states = target_mac_hidden_states.reshape(batch.batch_size, self.args.n_agents, batch.max_seq_length, -1).transpose(1,2) #btav

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, :] == 0] = -9999999  # From OG deepmarl
        mac_out_maxs = mac_out.clone()
        mac_out_maxs[avail_actions == 0] = -9999999

        # Best joint action computed by target agents
        target_max_actions = target_mac_out.max(dim=3, keepdim=True)[1]
        # Best joint-action computed by regular agents
        max_actions_qvals, max_actions_current = mac_out_maxs[:, :].max(dim=3, keepdim=True)

        if self.args.mixer == "qtran_base":
            # -- TD Loss --
            # Joint-action Q-Value estimates
            joint_qs, vs = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1])

            # Need to argmax across the target agents' actions to compute target joint-action Q-Values
            if self.args.double_q:
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device)
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
                max_actions_onehot = max_actions_current_onehot
            else:
                max_actions = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device)
                max_actions_onehot = max_actions.scatter(3, target_max_actions[:, :], 1)
            target_joint_qs, target_vs = self.target_mixer(batch[:, 1:], hidden_states=target_mac_hidden_states[:,1:], actions=max_actions_onehot[:,1:])

            # Td loss targets
            td_targets = rewards.reshape(-1,1) + self.args.gamma * (1 - terminated.reshape(-1, 1)) * target_joint_qs
            td_error = (joint_qs - td_targets.detach())
            masked_td_error = td_error * mask.reshape(-1, 1)
            td_loss = (masked_td_error ** 2).sum() / mask.sum()
            # -- TD Loss --

            # -- Opt Loss --
            # Argmax across the current agents' actions
            if not self.args.double_q: # Already computed if we're doing double Q-Learning
                max_actions_current_ = th.zeros(size=(batch.batch_size, batch.max_seq_length, self.args.n_agents, self.args.n_actions), device=batch.device )
                max_actions_current_onehot = max_actions_current_.scatter(3, max_actions_current[:, :], 1)
            max_joint_qs, _ = self.mixer(batch[:, :-1], mac_hidden_states[:,:-1], actions=max_actions_current_onehot[:,:-1]) # Don't use the target network and target agent max actions as per author's email

            # max_actions_qvals = th.gather(mac_out[:, :-1], dim=3, index=max_actions_current[:,:-1])
            opt_error = max_actions_qvals[:,:-1].sum(dim=2).reshape(-1, 1) - max_joint_qs.detach() + vs
            masked_opt_error = opt_error * mask.reshape(-1, 1)
            opt_loss = (masked_opt_error ** 2).sum() / mask.sum()
            # -- Opt Loss --

            # -- Nopt Loss --
            # target_joint_qs, _ = self.target_mixer(batch[:, :-1])
            nopt_values = chosen_action_qvals.sum(dim=2).reshape(-1, 1) - joint_qs.detach() + vs # Don't use target networks here either
            nopt_error = nopt_values.clamp(max=0)
            masked_nopt_error = nopt_error * mask.reshape(-1, 1)
            nopt_loss = (masked_nopt_error ** 2).sum() / mask.sum()
            # -- Nopt loss --

        elif self.args.mixer == "qtran_alt":
            raise Exception("Not supported yet.")

        loss = td_loss + self.args.opt_loss * opt_loss + self.args.nopt_min_loss * nopt_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            self.logger.log_stat("opt_loss", opt_loss.item(), t_env)
            self.logger.log_stat("nopt_loss", nopt_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            if self.args.mixer == "qtran_base":
                mask_elems = mask.sum().item()
                self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_targets", ((masked_td_error).sum().item()/mask_elems), t_env)
                self.logger.log_stat("td_chosen_qs", (joint_qs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("v_mean", (vs.sum().item()/mask_elems), t_env)
                self.logger.log_stat("agent_indiv_qs", ((chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents)), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
