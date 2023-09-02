import os
import cv2
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# 定义模型类（这是你已经训练好的模型）
class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
        # 在这里定义你的模型结构
        self.Conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(16 * 16 * 256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 2))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 16 * 16 * 256)
        x = self.Classes(x)
        return x

# 加载训练好的模型权重
model = Models()
model = torch.load('E:\\JetBrains\\PyCharm\\Programms\\computer vision\\modles\\Sep1_23\\01.pt')
model.to('cuda')  # 将模型移到GPU上

model.eval()  # 设置模型为评估模式

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # 调整图片大小，根据你的模型输入大小来选择
    transforms.ToTensor(),  # 将图片转换为PyTorch张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 根据你的训练数据的均值和标准差进行归一化
])

tensor_to_image = transforms.ToPILImage()

# 图片文件夹路径，替换为包含待预测图片的文件夹路径
image_folder = 'E:\\CoderLife\\demo\\DogsVSCats\\val\\Dog'

# 遍历文件夹中的图片
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)  # 使用OpenCV读取图片
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
    image = Image.fromarray(image)  # 转换为PIL图像
    image = transform(image)  # 应用数据变换
    image = image.unsqueeze(0)  # 添加批次维度
    print(type(image))
    image = image.to('cuda')
    with torch.no_grad():
        output = model(image)
    probabilities = F.softmax(output, dim=1)  # 使用softmax获取分类概率
    _, predicted_class = torch.max(probabilities, 1)  # 获取预测类别
    predicted_class = predicted_class.item()

    # 打印预测结果
    if(predicted_class == 0):
        print(f'Image: {image_name}, Predicted Class: It''s a Cat!')
        # image.show()
        img_3d = image[0]
        image_nottensor = tensor_to_image(img_3d)
        plt.imshow(image_nottensor)
        plt.title('pre: Cat')
        plt.show()
    else:
        print(f'Image: {image_name}, Predicted Class: It''s a Dog!')
        # image.show()
        img_3d = image[0]
        image_nottensor = tensor_to_image(img_3d)
        plt.imshow(image_nottensor)
        plt.title('pre: Dog')
        plt.show()
