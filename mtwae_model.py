import torch
import torch.nn as nn
import torch.nn.functional as F

class MTWAE(nn.Module):

    def __init__(self, in_features, latent_size):
        super(MTWAE, self).__init__()

        self.encoder_layer = nn.Sequential(
            nn.Linear(in_features, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 48),
            nn.LayerNorm(48),
            nn.LeakyReLU(),
            nn.Linear(48, 30),
            nn.LayerNorm(30),
            nn.LeakyReLU(),
            nn.Linear(30, latent_size),
        )

        self.decoder_layer = nn.Sequential(
            nn.Linear(latent_size, 30),
            nn.LayerNorm(30),
            nn.LeakyReLU(),
            nn.Linear(30, 48),
            nn.LayerNorm(48),
            nn.LeakyReLU(),
            nn.Linear(48, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, in_features),
        )

        self.Predicted_Bs_layer = nn.Sequential(
            nn.Linear(latent_size, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 1),
        )

        self.Predicted_Hc_layer = nn.Sequential(
            nn.Linear(latent_size, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 1),
        )

        self.Predicted_Dc_layer = nn.Sequential(
            nn.Linear(latent_size, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 90),
            nn.LayerNorm(90),
            nn.LeakyReLU(),
            nn.Linear(90, 1),
        )

    def encoder(self, X):

        z = self.encoder_layer(X)

        return z

    def decoder(self, z):

        x_reconst = F.softmax(self.decoder_layer(z),dim=1)

        return x_reconst
    
    def Predict_Bs(self, z):
    
        pre_Bs = self.Predicted_Bs_layer(z)

        return pre_Bs
    
    def Predict_Hc(self, z):
    
        pre_Hc = self.Predicted_Hc_layer(z)

        return pre_Hc
    
    def Predict_Dc(self, z):
    
        pre_Dc = self.Predicted_Dc_layer(z)

        return pre_Dc

    def forward(self, X):

        z = self.encoder(X)

        x_reconst = self.decoder(z)

        pro_Bs = self.Predict_Bs(z)

        pro_Hc = self.Predict_Hc(z)

        pro_Dc = self.Predict_Dc(z)

        return x_reconst, z, pro_Bs, pro_Hc, pro_Dc
