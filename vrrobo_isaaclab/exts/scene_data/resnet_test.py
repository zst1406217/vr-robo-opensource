import torch
from PIL import Image

import torchvision.models as models
import torchvision.transforms as transforms
import timm

# Load the ResNet-18 model
# model = getattr(models, "mobilenet_v3_small")(pretrained=True)
model=timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.eval()
model.cuda()
print(model)
# print(model.classifier.in_features)

# model.avgpool = torch.nn.AdaptiveMaxPool2d(2)
# model.heads = torch.nn.Identity()
# model.classifier=torch.nn.Identity()  # Remove the final fully connected layer
# max_pool = torch.nn.AdaptiveMaxPool2d(32)
# max_pool = torch.nn.AdaptiveAvgPool2d(1)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img_path = './obs_test.png'
img = Image.open(img_path)
img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
# print(img_tensor.shape)

import time
# Extract features
start=time.time()
img_tensor = img_tensor.cuda()
features = model(img_tensor)
with torch.no_grad():
    for i in range(100):
        features = model(img_tensor)
    # features = max_pool(features)
end=time.time()
print(f"Time taken: {end-start}")

torch.set_printoptions(profile="full")
print(features.shape)