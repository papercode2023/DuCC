import copy, time
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
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
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
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

                supcontrast_loss, p = self.supconLoss(anchor, positives, negatives)
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


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None, buffer=None):

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
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

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
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
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
