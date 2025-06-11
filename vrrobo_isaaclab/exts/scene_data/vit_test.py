import timm

model=timm.create_model('vit_tiny_patch16_224', pretrained=True)
model.eval()
config=timm.data.resolve_data_config({}, model=model)
transform=timm.data.create_transform(**config)
print(transform)
# print(model)