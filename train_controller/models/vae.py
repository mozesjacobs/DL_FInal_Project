import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, device, batch_size=250):
        super(VAE, self).__init__()

        self.device = device

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU()
        )

        self.mufc = nn.Linear(1024, 32)
        self.logvarfc = nn.Linear(1024, 32)

        self.decoder_fc = nn.Linear(32, 1024)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 128, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, stride=2),
            nn.Sigmoid(),
        )

        self.batch_size = batch_size
        # self.dist = torch.distributions.laplace.Laplace(0, torch.ones([50]))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        # noise = torch.randn(self.batch_size, 32).to(self.device)
        noise = torch.randn_like(std).to(self.device)
        return mu + std * noise  # z

    def forward(self, x):
        x = self.encoder(x)
        x = x.reshape(-1, 1024)
        mu, logvar = self.mufc(x), self.logvarfc(x)
        z = self.reparameterize(mu, logvar)
        z_ = self.decoder_fc(z)
        z_ = z_.reshape(-1, 1024, 1, 1)
        return self.decoder(z_.float()), mu, logvar

    def get_z(self, x):
        with torch.no_grad():
            encoded = self.encoder(x).reshape(-1, 1024)
            mu, logvar = self.mufc(encoded), self.logvarfc(encoded)
            return self.reparameterize(mu, logvar)

    def loss_func(self, x, x_prime, mu, logvar):
        recon_loss = nn.BCELoss(reduction='sum')
        loss = recon_loss(x_prime, x)
        loss += -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar))

        return loss