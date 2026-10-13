from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################################
#
# Policy Network
#
################################################################################


class Actor(nn.Module):
    """
    The policy network
    """

    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_action),
            nn.Tanh(),
        )

    def forward(self, s, g):
        x = torch.cat([s, g], -1)
        actions = self.max_action * self.net(x)
        return actions


################################################################################
#
# Critic Networks
#
################################################################################


class CriticMonolithic(nn.Module):
    """
    Monolithic Action-value Function Network (Q)
    """

    def __init__(self, args):
        super(CriticMonolithic, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal

        self.net = nn.Sequential(
            nn.Linear(dim_state + dim_goal + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, 1),
        )

    def forward(self, s, a, g):
        x = torch.cat([s, a / self.max_action, g], -1)
        q_value = self.net(x)
        return q_value


class CriticBilinear(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticBilinear, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed

        self.f = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        ff = self.f(x1)
        pp = self.phi(x2)
        q_value = ff.unsqueeze(1).bmm(pp.unsqueeze(-1)).view(-1, 1)
        return q_value


class CriticL2(nn.Module):
    """
    L2-norm Action-value Function
    Q(s, a, g) = ||f(s, a) - phi(s, g)||^2
    """

    def __init__(self, args):
        super(CriticL2, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed

        self.f = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        ff = self.f(x1)
        pp = self.phi(g)
        q_value = -(ff - pp).pow(2).mean(-1, keepdims=True)
        return q_value


class CriticAsym(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsym, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return -dist


class CriticAsymNew(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsymNew, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_new_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed
        self.gamma = args.gamma

        self.rew = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_goal, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x3 = torch.cat([s, a / self.max_action, g], -1)
        r = self.rew(x3)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return r - dist

    def sep_forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x3 = torch.cat([s, a / self.max_action, g], -1)
        r = self.rew(x3)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return r.detach() - dist, r


class CriticSym(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticSym, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        return -dist_s


class CriticSoftmax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticSoftmax, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        return -dist_a


class CriticMax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticMax, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, 450),
            nn.ReLU(inplace=True),
            nn.Linear(450, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        res = F.relu(asym1 - asym2)
        # dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist_a = res.max(-1)[0].view(-1, 1)
        return -dist_a


class CriticAsymMax(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsymMax, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        # A bug fix, since the metric should be the l2 norm instead of the square loss.
        # Thanks Wang et al for pointing this out at https://arxiv.org/abs/2211.15120
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True).sqrt()
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist


class CriticAsymMaxSAG(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsymMaxSAG, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, a / self.max_action, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist


class CriticAsymMaxSAGLatent(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsymMaxSAGLatent, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        dh = 140
        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dh),
            nn.ReLU(inplace=True),
        )
        self.g_emb = nn.Linear(dim_goal, dh)
        self.phi_emb = nn.Sequential(
            nn.Linear(dh * 2, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dh),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dh, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dh, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        # x2 = torch.cat([s, a/self.max_action, g], -1)
        fh = self.f_emb(x1)
        x2 = torch.cat([fh, self.g_emb(g)], -1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = res.max(-1)[0].view(-1, 1)
        dist = dist_s + dist_a
        return -dist


class CriticAsymLSE(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticAsymLSE, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.sym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        fh = self.f_emb(x1)
        phih = self.phi_emb(x2)

        sym1 = self.sym(fh)
        sym2 = self.sym(phih)
        asym1 = self.asym(fh)
        asym2 = self.asym(phih)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        # dist_a = res.max(-1)[0].view(-1, 1)
        dist_a = (res.exp().sum(-1, keepdims=True) + 1e-4).log()
        dist = dist_s + dist_a
        return -dist


class CriticModel(nn.Module):
    """
    Bilinear Action-value Function Network
    Q(s, a, g) = f(s, a)^T phi(s, g)

    Link: https://openreview.net/pdf?id=LedObtLmCjS
    """

    def __init__(self, args):
        super(CriticModel, self).__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_model_hidden = args.dim_model_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed
        self.dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_state),
        )

        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_model_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_model_hidden, dim_state),
        )

        self.sym = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )
        self.asym = nn.Sequential(
            nn.Linear(dim_state, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

    def forward(self, ns, gs):
        sym1 = self.sym(ns)
        sym2 = self.sym(gs)
        asym1 = self.asym(ns)
        asym2 = self.asym(gs)
        dist_s = (sym1 - sym2).pow(2).mean(-1, keepdims=True)
        res = F.relu(asym1 - asym2)
        dist_a = (F.softmax(res, -1) * res).sum(-1, keepdims=True)
        dist = dist_s + dist_a
        return -dist

    def forward_model(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        ns = self.f_emb(x1) + s
        gs = self.phi_emb(x2)
        return ns, gs


################################################################################
#
# Deep Norm Critic
#
################################################################################


class ConstrainedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, torch.min(self.weight**2, torch.abs(self.weight)))


class MaxAvgGlobalActivation(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = nn.Parameter(-torch.ones(1))
        if args.cuda:
            self.alpha.cuda()

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        return alpha * x.max(dim=-1)[0] + (1 - alpha) * x.mean(dim=-1)


class ConcaveActivation(nn.Module):
    def __init__(self, num_features, concave_activation_size, args):
        super().__init__()
        assert concave_activation_size > 1

        self.bs_nonzero = nn.Parameter(
            1e-3 * torch.randn((1, num_features, concave_activation_size - 1)) - 1
        )
        self.bs_zero = torch.zeros((1, num_features, 1))
        self.ms = nn.Parameter(
            1e-3 * torch.randn((1, num_features, concave_activation_size))
        )

        if args.cuda:
            self.bs_nonzero.cuda()
            self.bs_zero = self.bs_zero.cuda()
            self.ms.cuda()

    def forward(self, x):
        bs = torch.cat((F.softplus(self.bs_nonzero), self.bs_zero), -1)
        ms = 2 * torch.sigmoid(self.ms)
        x = x.unsqueeze(-1)

        x = x * ms + bs
        return x.min(-1)[0]


class ReduceMetric(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        if mode == "avg":
            self.forward = self.avg_forward
        elif mode == "max":
            self.forward = self.max_forward
        elif mode == "maxavg":
            self.maxavg_activation = MaxAvgGlobalActivation(args)
            self.forward = self.maxavg_forward
        elif mode == "softmax":
            self.forward = self.softmax_forward
        else:
            raise NotImplementedError

    def maxavg_forward(self, x):
        return self.maxavg_activation(x)

    def max_forward(self, x):
        return x.max(-1)[0]

    def avg_forward(self, x):
        return x.mean(-1)

    def softmax_forward(self, x):
        return (F.softmax(x, -1) * x).sum(-1)


class WideNormCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

        self.symmetric = False
        num_features = dim_embed
        self.num_components = 62
        concave_activation_size = 8
        if "softmax" in args.critic:
            mode = "softmax"
        elif "maxavg" in args.critic:
            mode = "maxavg"
        elif "max" in args.critic:
            mode = "max"
        else:
            mode = "avg"
        self.component_size = args.dim_embed

        output_size = self.component_size * self.num_components
        if not self.symmetric:
            num_features = num_features * 2
            self.f = ConstrainedLinear(num_features, output_size)
        else:
            self.f = nn.Linear(num_features, output_size)
        self.concave_activation = (
            ConcaveActivation(self.num_components, concave_activation_size, args)
            if concave_activation_size
            else nn.Identity()
        )
        self.reduce_metric = ReduceMetric(mode, args)

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x = self.phi(self.f_emb(x1))
        y = self.phi(self.phi_emb(x2))

        h = x - y
        if not self.symmetric:
            h = torch.cat((F.relu(h), F.relu(-h)), -1)
        h = torch.reshape(self.f(h), (-1, self.num_components, self.component_size))
        h = torch.norm(h, dim=-1)
        h = self.concave_activation(h)
        dist = self.reduce_metric(h).view(-1, 1)
        return -dist


class DeepNormCritic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_embed),
        )

        if "softmax" in args.critic:
            mode = "softmax"
        elif "maxavg" in args.critic:
            mode = "maxavg"
        elif "max" in args.critic:
            mode = "max"
        else:
            mode = "avg"
        layers = [162, 162]
        self.symmetric = False
        concave_activation_size = 8
        num_features = dim_embed
        self.Us = nn.ModuleList([nn.Linear(num_features, layers[0], bias=False)])
        self.Ws = nn.ModuleList([])

        for in_features, out_features in zip(layers[:-1], layers[1:]):
            self.Us.append(nn.Linear(num_features, out_features, bias=False))
            self.Ws.append(ConstrainedLinear(in_features, out_features, bias=False))

        self.activation = nn.ReLU()
        self.concave_activation = (
            ConcaveActivation(layers[-1], concave_activation_size, args)
            if concave_activation_size
            else nn.Identity()
        )
        self.reduce_metric = ReduceMetric(mode, args)

    def _asym_fwd(self, h):
        h1 = self.Us[0](h)
        for U, W in zip(self.Us[1:], self.Ws):
            h1 = self.activation(W(h1) + U(h))
        return h1

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        x = self.phi(self.f_emb(x1))
        y = self.phi(self.phi_emb(x2))

        h = x - y
        if self.symmetric:
            h = self._asym_fwd(h) + self._asym_fwd(-h)
        else:
            h = self._asym_fwd(h)
        h = self.concave_activation(h)
        dist = self.reduce_metric(h).view(-1, 1)
        return -dist


################################################################################
#
# PQE https://openreview.net/pdf?id=y0VvIg25yk
#
################################################################################

# from pqe import PQE
from src.pqe import PQE


class CriticPQE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_action = args.max_action
        dim_state = args.dim_state
        dim_hidden = args.dim_critic_hidden
        dim_action = args.dim_action
        dim_goal = args.dim_goal
        dim_embed = args.dim_embed

        self.discounted_pqe = PQE(
            num_quasipartition_mixtures=16, num_poisson_processes_per_quasipartition=4
        )

        self.f_emb = nn.Sequential(
            nn.Linear(dim_state + dim_action, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.phi_emb = nn.Sequential(
            nn.Linear(dim_state + dim_goal, dim_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(inplace=True),
        )
        self.encoder = nn.Sequential(
            nn.Linear(dim_hidden, 300), nn.ReLU(inplace=True), nn.Linear(300, 16 * 4)
        )

    def forward(self, s, a, g):
        x1 = torch.cat([s, a / self.max_action], -1)
        x2 = torch.cat([s, g], -1)
        zx = self.encoder(self.f_emb(x1))
        zy = self.encoder(self.phi_emb(x2))
        pred = self.discounted_pqe(zx, zy)
        return -pred.view(-1, 1)


# -------------------------
# Running normalizer (torch)
# -------------------------


@dataclass
class RunningNormTorch:
    """Online mean/std normalizer (like NormalizerNumpy).

    Stores running mean and variance using a numerically stable update.
    """

    size: int
    eps: float = 1e-4
    device: torch.device | None = None

    def __post_init__(self):
        dev = self.device if self.device is not None else torch.device("cpu")
        self.count = torch.tensor(0.0, device=dev)
        self.mean = torch.zeros(self.size, device=dev)
        self.m2 = torch.zeros(self.size, device=dev)  # sum of squared diffs

    @property
    def std(self) -> torch.Tensor:
        # avoid division by 0
        denom = torch.clamp(self.count, min=1.0)
        var = self.m2 / denom
        return torch.sqrt(var + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running moments with a batch x of shape (B, size)."""
        if x.numel() == 0:
            return
        x = x.detach()
        if x.dim() == 1:
            x = x.unsqueeze(0)

        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_m2 = batch_var * batch_count

        if self.count.item() == 0.0:
            self.mean = batch_mean
            self.m2 = batch_m2
            self.count = batch_count
            return

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / new_count)
        self.m2 = (
            self.m2 + batch_m2 + (delta**2) * (self.count * batch_count / new_count)
        )
        self.count = new_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# -------------------------
# Networks
# -------------------------


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Module:
    """Match the TF helper nn(): [hidden]*layers + [out_dim], ReLU between."""
    seq = []
    d = in_dim
    for _ in range(layers):
        seq.append(nn.Linear(d, hidden))
        seq.append(nn.ReLU(inplace=True))
        d = hidden
    seq.append(nn.Linear(d, out_dim))
    return nn.Sequential(*seq)


class _ForwardDynamics(nn.Module):
    """Single member: predicts delta in normalized state space."""

    def __init__(self, dimo: int, dimu: int, hidden: int = 256, layers: int = 4):
        super().__init__()
        self.dimo = dimo
        self.dimu = dimu
        self.net = _mlp(dimo + dimu, dimo, hidden=hidden, layers=layers)

    def forward(self, obs0_norm: torch.Tensor, act_norm: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs0_norm, act_norm], dim=-1)
        delta = self.net(x)
        return delta


# -------------------------
# Ensemble wrapper
# -------------------------


class EnsembleSingleStepDynamics(nn.Module):
    """Ensemble forward dynamics used by this repo's MHER agent.

    Required by call sites:
      - forward(s, a) -> next_s (same shape as s)
      - loss_fn(S, A, NS) -> scalar

    Notes:
      - The repo already normalizes observations/goals for the policy via
        `self._preproc_inputs`. This class additionally keeps its own
        normalizers (like the original MHER code). If the incoming S/NS are
        already normalized, these normalizers quickly converge to ~0 mean and
        ~1 std, so behavior remains sane.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Dimension discovery: adapt to common naming conventions.
        # (These are frequently present in HER/DDPG codebases.)
        if hasattr(args, "dim_state"):
            self.dimo = int(args.dim_state)
        elif hasattr(args, "state_dim"):
            self.dimo = int(args.state_dim)
        elif hasattr(args, "obs_dim"):
            self.dimo = int(args.obs_dim)
        else:
            raise AttributeError("args must define dim_state/state_dim/obs_dim")

        if hasattr(args, "dim_action"):
            self.dimu = int(args.dim_action)
        elif hasattr(args, "action_dim"):
            self.dimu = int(args.action_dim)
        elif hasattr(args, "act_dim"):
            self.dimu = int(args.act_dim)
        else:
            raise AttributeError("args must define dim_action/action_dim/act_dim")

        # Hyperparams (with sensible defaults matching the TF reference)
        self.num_models = int(
            getattr(
                args, "num_dynamics_models", getattr(args, "dynamics_ensemble_size", 5)
            )
        )
        self.hidden = int(getattr(args, "dynamics_hidden", 256))
        self.layers = int(getattr(args, "dynamics_layers", 4))
        self.norm_eps = float(getattr(args, "dynamics_norm_eps", 1e-4))
        self.clip_norm = float(getattr(args, "dynamics_clip_norm", 5.0))

        # Update mode: 'random' (bootstrapped) matches TF default.
        self.update_mode = getattr(args, "dynamics_update_mode", "random")
        # Prediction mode for forward(): default mean.
        self.pred_mode = getattr(args, "dynamics_pred_mode", "mean")

        # Per-ensemble normalizers (shared across members, like TF code)
        self.obs_normalizer = RunningNormTorch(self.dimo, eps=self.norm_eps)
        self.action_normalizer = RunningNormTorch(self.dimu, eps=self.norm_eps)

        self.models = nn.ModuleList(
            [
                _ForwardDynamics(
                    self.dimo, self.dimu, hidden=self.hidden, layers=self.layers
                )
                for _ in range(self.num_models)
            ]
        )

    # ----- utilities -----

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(0) if x.dim() == 1 else x

    @torch.no_grad()
    def _update_normalizers(
        self, obs0: torch.Tensor, actions: torch.Tensor, obs1: torch.Tensor
    ) -> None:
        self.obs_normalizer.update(obs0)
        self.obs_normalizer.update(obs1)
        self.action_normalizer.update(actions)

    def _normalize_inputs(
        self, obs0: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs0 = self._ensure_2d(obs0)
        actions = self._ensure_2d(actions)
        obs0_norm = self.obs_normalizer.normalize(obs0)
        act_norm = self.action_normalizer.normalize(actions)
        return obs0_norm, act_norm

    # ----- public API -----

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict next state (same space as s).

        Mirrors EnsembleForwardDynamics.predict_next_state(...):
          - mode='mean' -> ensemble mean
          - mode='random' -> one random member

        In this repo, forward() is used inside imaginary rollouts.
        """
        s = self._ensure_2d(s)
        a = self._ensure_2d(a)

        obs0_norm, act_norm = self._normalize_inputs(s, a)

        if self.pred_mode == "random":
            idx = int(torch.randint(low=0, high=self.num_models, size=(1,)).item())
            delta = self.models[idx](obs0_norm, act_norm)
            next_norm = obs0_norm + delta
        elif self.pred_mode == "mean":
            deltas = torch.stack(
                [m(obs0_norm, act_norm) for m in self.models], dim=0
            )  # (E,B,D)
            next_norm = obs0_norm + deltas.mean(dim=0)
        else:
            raise NotImplementedError(f"No such prediction mode: {self.pred_mode}")

        # Important: we *return in the same space as s*.
        # The TF reference denormalizes using obs_normalizer. Here, s is already
        # in whatever space the agent operates in (often normalized by the
        # agent's own normalizer). Returning next_norm keeps rollouts consistent.
        # If you ever need denormalized predictions, use:
        #   self.obs_normalizer.denormalize(next_norm)
        return next_norm

    def predict_next_state(
        self, obs0: torch.Tensor, actions: torch.Tensor, mode: str = "mean"
    ):
        """Convenience method mirroring TF API. Returns next-state in the same space as obs0."""
        old = self.pred_mode
        self.pred_mode = mode
        out = self.forward(obs0, actions)
        self.pred_mode = old
        return out

    def predict_next_state_mean_std(
        self, obs0: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, scalar std) similarly to TF 'mean_std' mode."""
        obs0 = self._ensure_2d(obs0)
        actions = self._ensure_2d(actions)
        obs0_norm, act_norm = self._normalize_inputs(obs0, actions)
        deltas = torch.stack(
            [m(obs0_norm, act_norm) for m in self.models], dim=0
        )  # (E,B,D)
        next_all = obs0_norm.unsqueeze(0) + deltas
        mean = next_all.mean(dim=0)
        # TF: std = result_array.std(axis=1).sum(axis=1)
        std = next_all.std(dim=0).sum(dim=1)  # (B,)
        return mean, std

    def loss_fn(
        self, S: torch.Tensor, A: torch.Tensor, NS: torch.Tensor
    ) -> torch.Tensor:
        """Train dynamics using TF-style MAE on delta.

        TF reference per-sample loss:
          mean(abs(pred_delta - obs1_norm + obs0_norm), axis=1)
        which is equivalent to MAE(pred_delta, (obs1_norm - obs0_norm)).

        Here we compute mean over batch and dims.
        """
        S = self._ensure_2d(S)
        A = self._ensure_2d(A)
        NS = self._ensure_2d(NS)

        # Update running stats *without* gradients (matches TF update())
        self._update_normalizers(S, A, NS)

        obs0_norm, act_norm = self._normalize_inputs(S, A)
        obs1_norm = self.obs_normalizer.normalize(NS)
        target_delta = (obs1_norm - obs0_norm).detach()

        if self.update_mode == "random":
            idx = int(torch.randint(low=0, high=self.num_models, size=(1,)).item())
            pred_delta = self.models[idx](obs0_norm, act_norm)
            loss = torch.mean(torch.abs(pred_delta - target_delta))
            return loss

        if self.update_mode == "all":
            losses = []
            for m in self.models:
                pred_delta = m(obs0_norm, act_norm)
                losses.append(torch.mean(torch.abs(pred_delta - target_delta)))
            return torch.stack(losses).mean()

        raise NotImplementedError(f"No such update mode: {self.update_mode}")

    def clip_gauss_noise(self, size: int) -> torch.Tensor:
        # TF reference returns 0; kept for parity / future use.
        return torch.zeros(size, device=self.obs_normalizer.mean.device)
