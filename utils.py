import torch

    
def awgn_channel(x):
    noise = torch.randn_like(x)
    return x + noise

def reparameterize(mu, ln_var):
    std = torch.exp(0.5 * ln_var)
    eps = torch.rand_like(std)
    c = mu + std * eps
    return c