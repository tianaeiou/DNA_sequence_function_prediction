import torch
import torch.nn as nn
import torch.nn.functional as F

class DanQ(nn.Module):
    def __init__(self, ):
        super(DanQ, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)
        self.Drop1 = nn.Dropout(p=0.2)
        self.BiLSTM = nn.LSTM(input_size=320, hidden_size=320, num_layers=2,
                              batch_first=True,
                              dropout=0.5,
                              bidirectional=True)
        self.Linear1 = nn.Linear(75 * 640, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x_x = torch.transpose(x, 1, 2)
        x, _ = self.BiLSTM(x_x)
        x = x.contiguous().view(-1, 75 * 640)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

class ConvFCNet(nn.Module):
    def __init__(self, ):
        super(ConvFCNet, self).__init__()
        self.Conv1 = nn.Conv1d(in_channels=4, out_channels=320, kernel_size=26)
        self.Maxpool = nn.MaxPool1d(kernel_size=13, stride=13)  # (bs,320,75)
        self.Drop1 = nn.Dropout(p=0.2)
        self.Linear1 = nn.Linear(75 * 320, 925)
        self.Linear2 = nn.Linear(925, 919)

    def forward(self, input):
        x = self.Conv1(input)
        x = F.relu(x)
        x = self.Maxpool(x)
        x = self.Drop1(x)
        x = x.contiguous().view(-1, 75 * 320)
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

def get_model(args):
    if args.model_name == "DanQ":
        return DanQ()
    elif args.model_name == "ConvFCNet":
        return ConvFCNet()
    else:
        raise NotImplementedError