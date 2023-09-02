from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "E:\\CoderLife\\demo\\supportfile\\hymenoptera_data\\train\\bees\\205835650_e6f2614bee.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')       # 注意 dataformats


# y=2*x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

writer.close()