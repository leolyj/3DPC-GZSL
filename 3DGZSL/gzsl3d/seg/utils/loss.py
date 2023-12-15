import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses:
    def __init__(
        self,
        weight=None,
        size_average=True,
        batch_average=True,
        ignore_index=-1,
        cuda=False, bs=4
    ):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda
        self.bs = bs

    def build_loss(self, mode="ce"):
        """Choices: ['ce' or 'focal']"""
        if mode == "ce":
            return self.CrossEntropyLoss
        elif mode == "focal":
            return self.FocalLoss
        elif mode == "ce_finetune":
            return self.CrossEntropyLossFinetune
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        if len(logit.size()) == 3:
            n, _, num_points = logit.size()
        elif len(logit.size()) == 2:
            n = self.bs
            _, classes = logit.size()
        else:
           n, _, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def CrossEntropyLossFinetune(self, logit, target):
        criterion = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index, size_average=self.size_average
        )
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= logit.shape[0]

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, _, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(
            weight=self.weight,
            ignore_index=self.ignore_index,
            size_average=self.size_average,
        )
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

class MSELoss:
    def __init__(self):
        pass
    def build_loss(self):
        return nn.MSELoss()

class GMMNLoss:
    def __init__(self, sigma=[2, 5, 10, 20, 40, 80], cuda=False):
        self.sigma = sigma
        self.cuda = cuda

    def build_loss(self):
        return self.moment_loss

    def get_scale_matrix(self, M, N):
        s1 = torch.ones((N, 1)) * 1.0 / N
        s2 = torch.ones((M, 1)) * -1.0 / M
        if self.cuda:
            s1, s2 = s1.cuda(), s2.cuda()
        return torch.cat((s1, s2), 0)

    def moment_loss(self, gen_samples, x):
        X = torch.cat((gen_samples, x), 0)
        XX = torch.matmul(X, X.t())
        X2 = torch.sum(X * X, 1, keepdim=True)
        exp = XX - 0.5 * X2 - 0.5 * X2.t()
        M = gen_samples.size()[0]
        N = x.size()[0]
        s = self.get_scale_matrix(M, N)
        S = torch.matmul(s, s.t())

        loss = 0
        for v in self.sigma:
            kernel_val = torch.exp(exp / v)
            loss += torch.sum(S * kernel_val)

        loss = torch.sqrt(loss)
        return loss

class InfoNCELosses(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_key, negative_keys=None):
        return self.info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)

    def info_nce(self, query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        # Check input dimensionality.
        if query.dim() != 2:
            raise ValueError('<query> must have 2 dimensions.')
        if positive_key.dim() != 2:
            raise ValueError('<positive_key> must have 2 dimensions.')
        if negative_keys is not None:
            if negative_mode == 'unpaired' and negative_keys.dim() != 2:
                raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
            if negative_mode == 'paired' and negative_keys.dim() != 3:
                raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

        # Check matching number of samples.
        if len(query) != len(positive_key):
            raise ValueError('<query> and <positive_key> must must have the same number of samples.')
        if negative_keys is not None:
            if negative_mode == 'paired' and len(query) != len(negative_keys):
                raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

        # Embedding vectors should have same number of components.
        if query.shape[-1] != positive_key.shape[-1]:
            raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
        if negative_keys is not None:
            if query.shape[-1] != negative_keys.shape[-1]:
                raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

        # Normalize to unit vectors
        query, positive_key, negative_keys = self.normalize(query, positive_key, negative_keys)
        if negative_keys is not None:
            # Explicit negative keys

            # Cosine between positive pairs
            positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

            if negative_mode == 'unpaired':
                # Cosine between all query-negative combinations
                negative_logits = query @ self.transpose(negative_keys)

            elif negative_mode == 'paired':
                query = query.unsqueeze(1)
                negative_logits = query @ self.transpose(negative_keys)
                negative_logits = negative_logits.squeeze(1)

            # First index in last dimension are the positive samples
            logits = torch.cat([positive_logit, negative_logits], dim=1)
            labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
        else:
            # Negative keys are implicitly off-diagonal positive keys.

            # Cosine between all combinations
            logits = query @ transpose(positive_key)

            # Positive keys are the entries on the diagonal
            labels = torch.arange(len(query), device=query.device)

        return F.cross_entropy(logits / temperature, labels, reduction=reduction)

    def transpose(self, x):
        return x.transpose(-2, -1)

    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

