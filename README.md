# test
# clip基本的原理
一种基于视觉和语言相互关联的图像分类模型CLIP模型的核心思想是将视觉和语言的表示方式相互联系起来，从而实现图像分类任务。具体来说，CLIP模型采用了对比学习和预训练的方法，使得模型能够在大规模无标注数据上进行训练，并学习到具有良好泛化能力的特征表示。
# 所使用的图片
![image](https://user-images.githubusercontent.com/128159015/227698797-3ed2d024-86d5-4d49-b35d-3e22dd1b7cfc.png)
# 不同类别的影响
首先设置'red','China','envelop'三种不同类型
```python
import os
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

image = preprocess(Image.open("img.png")).unsqueeze(0).to(device)

a_list=['red', 'China', 'envelop']
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in a_list ]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(3)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
```print(f"{a_list[index]:>16s}: { 100*value.item():.2f}%")
    
# 最后输出结果为
![image](https://user-images.githubusercontent.com/128159015/227699120-03a3adda-a936-4995-b643-2f4fc45f384e.png)

最终China的结果为最大

# 增加一个类别'Red envelope'
```python
import os
import clip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

image = preprocess(Image.open("img.png")).unsqueeze(0).to(device)

a_list=['red', 'China', 'envelop', 'red envelop']
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in a_list ]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_inputs)

image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
values, indices = similarity[0].topk(4)

print("\nTop predictions:\n")
for value, index in zip(values, indices):
```print(f"{a_list[index]:>16s}: { 100*value.item():.2f}%")
    
最后输出结果为
![image](https://user-images.githubusercontent.com/128159015/227699269-ca745e89-cdb1-4257-bde2-90cd1afd1c9e.png)
出现了一边倒相信'Red envelop'的情况
# 总结
当类别为'red','China','envelop'时经预测图片最为符合'China'，但当类别为'red','China','envelop','Red envelop'时'Red envelop'对该图片的描述更为一致，出现了一边倒的情况。所以当类别描述最为接近图片时出现的概率可能是最大的。
   
