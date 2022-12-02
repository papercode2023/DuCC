import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, anchor, pos , neg):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if anchor.is_cuda
                  else torch.device('cpu'))

        anchor, pos, neg = normalize(anchor, pos, neg)

        batch_size = anchor.shape[0]

        contrast_count = pos.shape[1]
        contrast_feature = pos.permute(1, 2, 0).contiguous()

        anchor_feature = anchor.unsqueeze(0)
     
        neg_feature = neg.permute(1, 2, 0).contiguous()


        # compute pos logits
        anchor_dot_contrast_pos = torch.div(
            torch.matmul(anchor_feature, contrast_feature),
            self.temperature)
        # for numerical stability
        logits_max_pos, _ = torch.max(anchor_dot_contrast_pos, dim=2, keepdim=True)
        logits_pos = anchor_dot_contrast_pos - logits_max_pos.detach()

        # compute neg logits
        anchor_dot_contrast_neg = torch.div(
            torch.matmul(anchor_feature, neg_feature),
            self.temperature)
        # for numerical stability
        logits_max_neg, _ = torch.max(anchor_dot_contrast_neg, dim=2, keepdim=True)
        logits_neg = anchor_dot_contrast_neg - logits_max_neg.detach()


        # compute log_prob
        exp_logits = torch.exp(logits_neg)
        log_prob = logits_pos - torch.log(exp_logits.sum(2, keepdim=True).sum(0, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob.mean(2)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        p = loss.mean(0).detach().cpu().numpy()

        loss = loss.view(contrast_count, batch_size).mean()

        return loss, p

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]