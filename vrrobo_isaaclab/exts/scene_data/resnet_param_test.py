from torchvision import models

cnn_model = models.mobilenet_v3_small(pretrained=True)
weights=models.get_weight("MobileNet_V3_Small_Weights.IMAGENET1K_V1")
print(weights.transforms)

# print(cnn_model)
