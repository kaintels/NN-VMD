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

class TaskLayer(nn.Module):
    def __init__(self):
        super(TaskLayer, self).__init__()

        self.conv1 = nn.Conv1d(1, 32, 3, 1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, 3, 1, padding=1)
        self.conv3 = nn.Conv1d(64, 128, 3, 1, padding=1)
        self.flatten = nn.Flatten()
        self.maxpool = nn.MaxPool1d(2)
        self.decomp1 = nn.Linear(2176, 1000)
        self.decomp2 = nn.Linear(1000, 420)
        self.classify1 = nn.Linear(2176, 300)
        self.classify2 = nn.Linear(300, 5)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = torch.relu(x)
        x = self.maxpool(self.conv2(x))
        x = torch.relu(x)
        x = self.maxpool(self.conv3(x))
        x = torch.relu(x)
        x = self.flatten(x)
        class_x = self.classify1(x)
        class_x = self.classify2(class_x)
        decomp_x = self.decomp1(x)
        decomp_x = self.decomp2(decomp_x)
        decomp_x = decomp_x.reshape(-1, 3, 140)

        return decomp_x, class_x


if __name__ == "__main__":
    dummy = torch.rand((1, 1, 140))
    model = TaskLayer()

    print(model(dummy))
    torch.onnx.export(model, dummy, "./models/model.onnx", input_names=["signals"], output_names=["reconst_signals", "classification"])





