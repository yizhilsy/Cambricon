import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# python用法 -》 tensor数据类型
# 使用神经网络需要tensor类

img_path = "D:\\FILE\\dogpicture\\path13.jpg"

img = Image.open(img_path)
img.show()
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)
print(type(tensor_img))

writer = SummaryWriter("logs")
writer.add_image("Tensor_img", tensor_img)
writer.close()
cv_img = cv2.imread(img_path)
print(type(cv_img))

