import torch.nn as nn
import torch.nn.functional as F
import torch as th

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape + args.repre_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        half_repre_hidden = 128

        self.representation = nn.Sequential(
            nn.Linear(self.args.obs_shape, half_repre_hidden * 2),
            nn.ReLU(),
            nn.Linear(half_repre_hidden * 2, half_repre_hidden),
            nn.ReLU(),
            nn.Linear(half_repre_hidden, args.repre_dim)
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ind_obs):
        # representation
        with th.no_grad():
            repre = self.representation(ind_obs.contiguous().view(-1, self.args.obs_shape))

        # RNN
        inputs = th.cat((inputs, repre), dim=1)
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
