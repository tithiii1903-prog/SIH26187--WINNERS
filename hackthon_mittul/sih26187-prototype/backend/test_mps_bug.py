import torch
from facenet_pytorch import InceptionResnetV1

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")

resnet_cpu = InceptionResnetV1(pretrained='vggface2').eval().cpu()
resnet_mps = InceptionResnetV1(pretrained='vggface2').eval().to(device)

x = torch.randn(1, 3, 160, 160)
out_cpu = resnet_cpu(x)
out_mps = resnet_mps(x.to(device)).cpu()

sim = torch.nn.functional.cosine_similarity(out_cpu, out_mps).item()
print(f"Similarity between CPU and MPS for same input: {sim}")
