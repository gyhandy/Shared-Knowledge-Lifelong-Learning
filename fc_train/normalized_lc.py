import torch
import torch.nn as nn

class Normalize_FC(nn.Module):
    def __init__(self, num_classes, setting_bias=True, norm_usage=2):
        super(Normalize_FC,self).__init__()

        self.num_classes = num_classes
        self.setting_bias = setting_bias
        self.norm_usage = norm_usage
        
        self.fc = nn.Linear(2048, num_classes, bias=setting_bias)

    def forward(self, x):
        x = self.fc(x)
        weight = self.fc.weight
        if self.setting_bias:
            bias = self.fc.bias
            combined_matrix = torch.cat([weight, bias.unsqueeze(1)], dim=1)
        else:
            combined_matrix = weight

        if self.norm_usage != 0:
            norm = combined_matrix.norm(p=self.norm_usage, dim=1)
        else:
            norm, _ = torch.max(torch.abs(combined_matrix), dim=1)
        x = x/norm
        return x
    
if __name__ == "__main__":
    pass