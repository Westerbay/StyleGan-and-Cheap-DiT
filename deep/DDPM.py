import torch.nn as nn
import torch


class DDPM(nn.Module):

    def __init__(self, time_stemps, device):
        super().__init__()
        self.time_stemps = time_stemps
        self.device = device

        # 1e-4 variance minimale pour les premiers pas de temps
        # 0.02 variance maximale pour les derniers pas de temps
        betas = torch.linspace(1e-4, 0.02, time_stemps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    
    def forward_sample(self, x_0, t): # q_sample
        noise = torch.randn_like(x_0)
        sqrt_alpha_cumprod = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_cumprod * x_0 + sqrt_one_minus_alpha_cumprod * noise, noise
    
    @torch.no_grad() # p_sample
    def backward_sample(self, model, x_t, t):
        beta_t = self.betas[t]
        sqrt_one_minus_alpha_cumprod = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]

        eps_theta = model(x_t, t)
        model_mean = beta_t[:, None, None, None] * eps_theta
        model_mean = model_mean / sqrt_one_minus_alpha_cumprod[:, None, None, None]
        model_mean = x_t - model_mean
        model_mean = sqrt_recip_alpha[:, None, None, None] * model_mean

        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(beta_t)[:, None, None, None] * noise

    @torch.no_grad()
    def sample(self, model, img_size, batch_size):
        x = torch.randn(batch_size, 3, img_size, img_size, device=self.device)
        for i in reversed(range(self.time_stemps)):
            t = torch.ones((batch_size,), device=self.device).long() * i
            x = self.backward_sample(model, x, t)
        return x
