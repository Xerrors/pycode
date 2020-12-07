import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

data_dir = "./data"


def CelebA(batch_size=128, num_worker=8, resize=None):
    """ Site: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html """
    actions = [
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    if resize:
        actions.insert(0, transforms.Resize(resize))

    transform = transforms.Compose(actions)

    dataset = torchvision.datasets.ImageFolder(root=data_dir+"/celeba", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_worker)
    return dataloader