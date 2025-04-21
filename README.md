# 狗品种识别模型

这个基于ResNet+Transformer的深度学习模型可以识别约70种不同的狗品种。

## 模型文件

模型文件包含在本仓库中：
- `resnetTransformer_mobile.pt`: 主模型文件(PyTorch格式)

## 模型信息
- 架构: ResNet+Transformer
- 输入尺寸: 224x224 RGB图像
- 输出: 70种狗品种的分类结果

## 使用方法

```python
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = torch.load("resnetTransformer_mobile.pt")
model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像并预处理
image = Image.open("dog_image.jpg")
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 执行预测
with torch.no_grad():
    output = model(input_batch)
    
# 获取预测结果
_, predicted = torch.max(output, 1)
```

## 狗品种列表

模型可识别以下70种狗品种：
Afghan, African Wild Dog, Airedale, American Hairless, American Spaniel, Basenji, Basset, Beagle, Bearded Collie, Bermaise, Bichon Frise, Blenheim, Bloodhound, Bluetick, Border Collie, Borzoi, Boston Terrier, Boxer, Bull Mastiff, Bull Terrier, Bulldog, Cairn, Chihuahua, Chinese Crested, Chow, Clumber, Cockapoo, Cocker, Collie, Corgi, Coyote, Dalmation, Dhole, Dingo, Doberman, Elk Hound, French Bulldog, German Sheperd, Golden Retriever, Great Dane, Great Perenees, Greyhound, Groenendael, Irish Spaniel, Irish Wolfhound, Japanese Spaniel, Komondor, Labradoodle, Labrador, Lhasa, Malinois, Maltese, Mex Hairless, Newfoundland, Pekinese, Pit Bull, Pomeranian, Poodle, Pug, Rhodesian, Rottweiler, Saint Bernard, Schnauzer, Scotch Terrier, Shar_Pei, Shiba Inu, Shih-Tzu, Siberian Husky, Vizsla, Yorkie
