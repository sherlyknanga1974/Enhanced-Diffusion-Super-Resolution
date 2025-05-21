import torch
from torch import optim
from edsrm import EDSRM
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])
dataset = datasets.CIFAR10(root="./data", train=True, transform=transform, download=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = EDSRM().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(5):
    pbar = tqdm(loader)
    for x, _ in pbar:
        x = x.cuda()
        low_res = torch.nn.functional.interpolate(x, scale_factor=0.5, mode='bilinear')
        low_res_up = torch.nn.functional.interpolate(low_res, scale_factor=2, mode='bilinear')
        t = torch.randint(0, 999, (x.size(0),), device=x.device)
        output = model(low_res_up, t)
        loss = loss_fn(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch}, Loss: {loss.item():.4f}")