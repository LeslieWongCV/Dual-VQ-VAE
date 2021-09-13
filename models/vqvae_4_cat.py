
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder
from models.betavae_4_cat import BetaVAE_H

class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # MI tools 
        device = 'cuda:0'
        sigma=0.4
        num_bins=64
        normalize=True
        self.sigma = 2*sigma**2
        self.num_bins = num_bins
        self.normalize = normalize
        self.epsilon = 1e-10
        self.bins = nn.Parameter(torch.linspace(0, 255, num_bins, device=device).float(), requires_grad=False)

        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.encoder_beta = BetaVAE_H()
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        self.pre_quantization_conv_2 = nn.Conv2d(
            embedding_dim, 1, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        # self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.decoder = Decoder(embedding_dim+1, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False):
        
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)  # h_dim - embedding_dim 128 - 64
        embedding_loss, z_q, perplexity, _, _ = self.vector_quantization(
            z_e)
        z_beta, _, _, kld_loss = self.encoder_beta(x)  # z：torch.Size([32, 64, 8, 8])
        total_z_q = torch.cat((z_q, z_beta), dim=1)

        z_q = self.pre_quantization_conv_2(z_q)  # h_dim - embedding_dim 128 - 64
        MI = self.getMutualInformation(z_q, z_beta)
        x_hat = self.decoder(total_z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity, kld_loss, MI

    def marginalPdf(self, values):
        residuals = values - self.bins.unsqueeze(0).unsqueeze(0)
        kernel_values = torch.exp(-0.5*(residuals / self.sigma).pow(2))
        pdf = torch.mean(kernel_values, dim=1)
        normalization = torch.sum(pdf, dim=1).unsqueeze(1) + self.epsilon
        pdf = pdf / normalization
        return pdf, kernel_values
    
    def jointPdf(self, kernel_values1, kernel_values2):
        joint_kernel_values = torch.matmul(kernel_values1.transpose(1, 2), kernel_values2)
        normalization = torch.sum(joint_kernel_values, dim=(1,2)).view(-1, 1, 1) + self.epsilon
        pdf = joint_kernel_values / normalization
        return pdf
        
    def getMutualInformation(self, input1, input2):

        # Torch tensors for images between (0, 1)
        input1 = input1*255
        input2 = input2*255
        B, C, H, W = input1.shape
        assert((input1.shape == input2.shape))

        x1 = input1.view(B, H*W, C)
        x2 = input2.view(B, H*W, C)
        
        pdf_x1, kernel_values1 = self.marginalPdf(x1)
        pdf_x2, kernel_values2 = self.marginalPdf(x2)
        pdf_x1x2 = self.jointPdf(kernel_values1, kernel_values2)

        H_x1 = -torch.sum(pdf_x1*torch.log2(pdf_x1 + self.epsilon), dim=1)
        H_x2 = -torch.sum(pdf_x2*torch.log2(pdf_x2 + self.epsilon), dim=1)
        H_x1x2 = -torch.sum(pdf_x1x2*torch.log2(pdf_x1x2 + self.epsilon), dim=(1,2))

        mutual_information = H_x1 + H_x2 - H_x1x2
        
        if self.normalize:
            mutual_information = 2*mutual_information/(H_x1+H_x2)

        return mutual_information

