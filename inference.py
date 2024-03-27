import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F


model = torch.load('model.pth')
image = Image.open('image.jpg')
output = model.forward(image.unsqueeze(0))

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

image = transform(image)
probs = F.softmax(output,dim=1)
pred = torch.argmax(probs,dim=1)
print(pred)


