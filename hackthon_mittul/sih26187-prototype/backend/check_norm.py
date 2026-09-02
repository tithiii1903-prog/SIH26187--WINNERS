import torch
from facenet_pytorch import InceptionResnetV1

resnet = InceptionResnetV1(pretrained='vggface2').eval()
x = torch.randn(1, 3, 160, 160)
out = resnet(x)
print("Norm:", torch.norm(out).item())
