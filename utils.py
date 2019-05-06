import torch
import torch.nn as nn
import numpy as np

# def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
#     """
#     Sample random noise vector for training.
#     INPUT
#     --------
#     n_dis_c : Number of discrete latent code.
#     dis_c_dim : Dimension of discrete latent code.
#     n_con_c : Number of continuous latent code.
#     n_z : Dimension of inicompressible noise.
#     batch_size : Batch Size
#     device : GPU/CPU
#     """

#     z = torch.randn(batch_size, n_z, 1, 1, device=device)

#     idx = np.zeros((n_dis_c, batch_size))
#     if(n_dis_c != 0):
#         dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)
        
#         for i in range(n_dis_c):
#             idx[i] = np.random.randint(dis_c_dim, size=batch_size)
#             dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

#         dis_c = dis_c.view(batch_size, -1, 1, 1)

#     if(n_con_c != 0):
#         # Random uniform between -1 and 1.
#         con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

#     noise = z
#     if(n_dis_c != 0):
#         noise = torch.cat((noise, dis_c), dim=1)
#     if(n_con_c != 0):
#         noise = torch.cat((noise, con_c), dim=1)

#     return noise, idx


def noise_sample(dis_c_dim, n_con_c, n_z, batch_size, device):

    dis_c = torch.FloatTensor(batch_size, dis_c_dim).to(device)
    con_c = torch.FloatTensor(batch_size, n_con_c).to(device)
    noise = torch.FloatTensor(batch_size, n_z).to(device)

    idx = np.random.randint(dis_c_dim, size=batch_size)
    c = np.zeros((batch_size, dis_c_dim))
    c[range(batch_size), idx] = 1.0

    dis_c.data.copy_(torch.Tensor(c))
    con_c.data.uniform_(-1.0, 1.0)
    noise.data.uniform_(-1.0, 1.0)
    n_dim = dis_c_dim + n_con_c + n_z
    z = torch.cat([noise, dis_c, con_c], 1).view(-1, n_dim, 1, 1)

    return z, idx

class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised.
    Treating Q(cj | x) as a factored Gaussian.
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - (x - mu).pow(2).div(var.mul(2.0) + 1e-6)
        nll = -(logli.sum(1).mean())

        return nll

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)