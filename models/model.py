import torch
import torch.nn as nn

class VMDNet(nn.Module):
    def __init__(self):
        super(VMDNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, 1, padding="same")
        self.conv2 = nn.Conv1d(32, 64, 3, 1, padding="same")
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(8960, 420)
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.sigmoid(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = x.reshape(-1, 3, 140)
        return x


class VMD_VAE_DNN(nn.Module):
    def __init__(self):
        super(VMD_VAE_DNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(140, 20),
            nn.Sigmoid()
            )
        
        self.mu = nn.Linear(20, 5)
        self.log_var = nn.Linear(20, 5)

        self.decoder = nn.Sequential(
            nn.Linear(5, 20),
            nn.Sigmoid(),
            nn.Linear(20, 140)
        )

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epslion = torch.randn_like(std)

        return epslion.mul(std).add_(mu)

    def forward(self, x):
        x = self.encoder(x)
        mu, log_var = self.mu(x), self.log_var(x)
        z = self.sampling(mu, log_var)
        out = self.decoder(z)
        return out, mu, log_var

class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()

        self.layer1 = nn.Linear(140, 5)
        self.flatten = nn.Flatten()

    def forward(self, x):

        x = self.flatten(x)
        x = self.layer1(x)

        return x

if __name__ == "__main__":
    dummy = torch.rand((1, 140))
    model = VMD_VAE_DNN()

    print()
    torch.onnx.export(model, dummy, "./models/model.onnx", input_names=["signals"], output_names=["reconst_signals"])





