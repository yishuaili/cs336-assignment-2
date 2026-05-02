import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"- the model parameters within the autocast context: {self.fc1.weight.dtype}")
        
        fc1_out = self.fc1(x)
        print(f"- the output of the first feed-forward layer (ToyModel.fc1): {fc1_out.dtype}")
        
        x = self.relu(fc1_out)
        
        ln_out = self.ln(x)
        print(f"- the output of layer norm (ToyModel.ln): {ln_out.dtype}")
        
        logits = self.fc2(ln_out)
        print(f"- the model's predicted logits: {logits.dtype}")
        
        return logits

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ToyModel(5, 2).to(device)
    
    x = torch.randn(4, 5).to(device)
    target = torch.randint(0, 2, (4,)).to(device)
    criterion = nn.CrossEntropyLoss()
    
    with torch.autocast(device_type=device.type if device.type == 'cuda' else 'cpu', dtype=torch.float16):
        logits = model(x)
        loss = criterion(logits, target)
        print(f"- the loss: {loss.dtype}")
        
    loss.backward()
    
    print(f"- and the model's gradients (e.g. fc1.weight.grad): {model.fc1.weight.grad.dtype}")

if __name__ == "__main__":
    main()
