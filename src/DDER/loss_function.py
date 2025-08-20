import torch
import torch.nn.functional as F


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, useKL=True):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    if not useKL:
        return loglikelihood

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device, useKL=False):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    if not useKL:
        return A

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div

def kl_loss(alpha, y, epoch_num, num_classes, annealing_step, device):
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    # print(kl_div)
    return torch.mean(kl_div)

def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    return torch.mean(loss)


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.log, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)


def edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    loss = edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    return torch.mean(loss)



def get_dc_loss(evidences, device):
    num_views = len(evidences)
    batch_size, num_classes = evidences[0].shape[0], evidences[0].shape[1]
    p = torch.zeros((num_views, batch_size, num_classes)).to(device)
    u = torch.zeros((num_views, batch_size)).to(device)
    for v in range(num_views):
        alpha = evidences[v] + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        p[v] = alpha / S
        u[v] = torch.squeeze(num_classes / S)
    dc_sum = 0
    for i in range(num_views):
        pd = torch.sum(torch.abs(p - p[i]) / 2, dim=2) / (num_views - 1)  # (num_views, batch_size)
        cc = (1 - u[i]) * (1 - u)  # (num_views, batch_size)
        dc = pd * cc
        dc_sum = dc_sum + torch.sum(dc, dim=0)
    dc_sum = torch.mean(dc_sum)
    return dc_sum

def SD(evidences, device):
    alpha = evidences + 1
    S = torch.sum(alpha, dim=1, keepdim=True).to(device)
    b = evidences/(S.expand(evidences.shape)).to(device)
    # b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
    batch_size, classes =evidences.shape[0], evidences.shape[1]
    SD = torch.zeros((batch_size)).to(device)

    b = b.t()
    for i in range(classes):
        for j in range(classes):
            if(i!=j):
                new = 1 - torch.abs(torch.sub(b[i], b[j]))
                # print(new)
                SD = torch.add(SD, new.t())

    ans = torch.mean(SD)
    ans = torch.where(torch.isnan(ans), torch.zeros_like(ans), ans)
    # print(ans)

    return ans


def Diss(evidences, device):
    alpha = evidences + 1
    S = torch.sum(alpha, dim=1, keepdim=True).to(device)
    b = evidences/(S.expand(evidences.shape)).to(device)
    b = torch.where(torch.isnan(b), torch.zeros_like(b), b)
    # u = len(b[0])/(S.expand(evidences.shape)).to(device)
    # print(u)
    batch_size, classes =evidences.shape[0], evidences.shape[1]
    diss = torch.zeros((batch_size, classes)).to(device)

    # print(b[100])
    b = b.t()
    for k in range(classes):
        denominator = torch.clip(torch.sum(b, dim=0, keepdim=True) - b[k], min=1e-8)
        oth = b[k] / denominator
        numerator = torch.abs(b - b[k])
        denominator = b + b[k]
        denominator = torch.clip(denominator, min=1e-8)
        Bal = 1 - torch.nan_to_num(torch.div(numerator, denominator), 0)
        Bal = Bal.masked_fill(Bal == 1, 0)

        diss += (b * oth * Bal).t()

    ans = torch.mean(diss)
    ans = torch.where(torch.isnan(ans), torch.zeros_like(ans), ans)
    # print(ans)

    return ans


def get_loss(evidences, evidence_a, evidence_con, evidence_div, target, epoch_num, num_classes, annealing_step, gamma, delta, device):
    target = F.one_hot(target, num_classes)
    alpha_con = evidence_con + 1
    alpha_div = evidence_div + 1
    loss = 0
    for v in range(len(evidences)):
        alpha = evidences[v] + 1
        loss += edl_digamma_loss(alpha, target, epoch_num, num_classes, annealing_step, device)
    loss += edl_digamma_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)
    loss += gamma * kl_loss(alpha_div, target, epoch_num, num_classes, annealing_step, device)

    # loss_acc += gamma * edl_digamma_loss(alpha_div, target, epoch_num, num_classes, annealing_step, device)
    loss += delta * kl_loss(alpha_con, target, epoch_num, num_classes, annealing_step, device)
    if loss.isnan():
        print(loss)

    return loss
