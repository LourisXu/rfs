import numpy as np
import torch
from torchvision import models, transforms
import torchsummary


def myHook(self, input, output):
    print("enter hook")
    print(type(input))
    print(len(input))
    print(input[0].shape)
    print(output[0].shape)
    print("exit hook")


net = models.alexnet(pretrained=False, num_classes=2)
net.features[6].register_forward_hook(myHook)
x = torch.randn(10, 3, 128, 128)
print(net(x).shape)
print(net.state_dict().keys())


# x = np.random.uniform(size=(10, 128, 128, 3))
# aug = transforms.Compose(
#     [transforms.Resize(32),
#     transforms.ToTensor()]
# )
# y = aug(x)
# print(y.shape)
