from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("D:\\FILE\\dogpicture\\path18.jpg")
print(img)

trans_totensor = transforms.ToTensor()      # 赋值为一个ToTensor对象
img_tensor = trans_totensor(img)
print(type(img_tensor))
writer.add_image("ToTensor", img_tensor, 3)

# Normalize
# 原先
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1,0.3,5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm,3)


# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# 此时img_resize是一个PIL数据类型
print(img_resize)
# 将img_resize的数据类型转为tensor数据类型
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, 0)
print(img_resize)
# Compose -resize - 2
trans_resize_2 = transforms.Resize(512)
# Compose类中第二个
# PIL->PIL->tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)
writer.close()