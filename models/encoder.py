import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from models.custom import GRULayer, GRUCell, mvn


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, image_size):
        super(Encoder, self).__init__()
        height = image_size[0]
        width = image_size[1]
        self.convs = nn.Sequential(
            nn.Conv2d(input_dim, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2),
            nn.ReLU(inplace=True)
        )

        for i in range(4):
            width = (width - 1) // 2
            height = (height - 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(64 * width * height, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.shape[0], -1)
        return self.mlp(x)


class GaussianParamNet(nn.Module):
    """
    Parameterise a Gaussian distributions.
    """
    def __init__(self, input_dim, output_dim):
        super(GaussianParamNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim, bias=False)
        self.layer_nml = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.fc2 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        """
        x: input image with shape [B, K, 2*D]
        """
        # obtain size of []
        x = self.fc2(F.relu(self.layer_nml(self.fc1(x))))
        mu, sigma = x.chunk(2, dim=-1)
        sigma = F.softplus(sigma + 0.5) + 1e-8
        return mu, sigma


class IndependentPrior(nn.Module):

    def __init__(self, z_size):
        super(IndependentPrior, self).__init__()
        self.z_size = z_size
        self.z_linear = nn.Sequential(
            nn.Linear(self.z_size, 128),
            nn.ELU(True))
        self.z_mu = nn.Linear(128, self.z_size)
        self.z_softplus = nn.Linear(128, self.z_size)

    def forward(self, slots):
        """
        slots is [N,K,D]
        """
        slots = self.z_linear( slots )  # [N,K,D]
        loc_z = self.z_mu( slots )
        sp_z = self.z_softplus( slots )
        return loc_z, sp_z

class ObjectDiscoveryBlock(nn.Module):
    def __init__(self, K, z_size, input_size, stochastic_layers):
        super(ObjectDiscoveryBlock, self).__init__()
        self.K = K
        self.z_size = z_size
        self.num_stochastic_layers = stochastic_layers
        self.scale = z_size ** -0.5
        self.eps = 1e-8
        self.H, self.W = input_size
        self.C = 3

        # ODB INFERENCE PARAMS  
        self.norm_slots = nn.LayerNorm(self.z_size)
        self.norm_mu_pre_ff = nn.LayerNorm(self.z_size)
        self.norm_softplus_pre_ff = nn.LayerNorm(self.z_size)
         
        self.to_q = nn.Linear(self.z_size, self.z_size, bias = False)
        self.to_k = nn.Linear(64, self.z_size, bias = False)
        self.to_v = nn.Linear(64, self.z_size, bias = False)

        self.gru = GRULayer(GRUCell, self.z_size, self.z_size)

        #self.gru_mu = nn.GRU(self.z_size, self.z_size)
        self.mlp_mu = nn.Sequential(
                nn.Linear(self.z_size, self.z_size*2),
                nn.ReLU(True),
                nn.Linear(self.z_size*2, self.z_size)
        )
        #self.gru_softplus = nn.GRU(self.z_size, self.z_size)
        self.mlp_softplus = nn.Sequential(
                nn.Linear(self.z_size, 2 * self.z_size),
                nn.ReLU(True),
                nn.Linear(self.z_size * 2, self.z_size)
        )

        #self.init_posterior = nn.Parameter(torch.cat([torch.zeros(1,self.z_size), torch.ones(1,self.z_size)],1))

        self.generation_relation = IndependentPrior(self.z_size)
        self.indep_prior = self.generation_relation
    

    def forward(self, inputs, prev_lambda=None):
        x_locs, masks, posteriors = [], [], []
        all_samples = {}

        loc, sp = prev_lambda.chunk(2, dim=1)
        loc = loc.contiguous()
        sp = sp.contiguous()
        init_posterior = mvn(loc, sp)
        slots = init_posterior.rsample()
        loc_shape = loc.shape
        slots = slots.view(-1, self.K, self.z_size)  # [N, K, D]
        
        slots_mu = loc
        slots_softplus = sp

        k, v = self.to_k(inputs), self.to_v(inputs)
        
        for layer in range(self.num_stochastic_layers):  

            # scaled dot-product attention
            slots = self.norm_slots(slots)
            q = self.to_q(slots)
            #all_samples[f'posterior_z_{layer}_conditional'] = q            
            #dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            # TODO: this ordering change shouldn't matter..
            q *= self.scale
            dots = torch.einsum('bid,bjd->bij', q, k)
            # dots is [batch_size, num_slots, num_inputs]

            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)
            updates = torch.einsum('bjd,bij->bid', v, attn)
            updates_recurrent = torch.cat([updates, updates],2)
            slots_recurrent = torch.cat([slots_mu, slots_softplus],1)
            slots_recurrent, _ = self.gru(
                updates_recurrent.reshape(1,-1,2*self.z_size),
                slots_recurrent.reshape(-1,2*self.z_size))
            slots_mu, slots_softplus = slots_recurrent[0].chunk(2,1)
            slots_mu = slots_mu + self.mlp_mu(self.norm_mu_pre_ff(slots_mu))
            slots_softplus = slots_softplus + self.mlp_softplus(self.norm_softplus_pre_ff(slots_softplus))
            
            lamda = torch.cat([slots_mu, slots_softplus], 1)  # [N*K, 2*z_size]
            slots_mu, slots_softplus = lamda.chunk(2,1)

            # Sample
            posterior_z = mvn(slots_mu, slots_softplus)
            slots = posterior_z.rsample()

            posteriors += [posterior_z]
            all_samples[f'posterior_z_{layer}'] = slots.view(-1, self.K, self.z_size)
                            
            if layer == self.num_stochastic_layers-1:
                continue

            slots = slots.view(-1, self.K, self.z_size)

        # decode
        slots = slots.view(-1, self.z_size)
        return slots, posteriors, all_samples, lamda