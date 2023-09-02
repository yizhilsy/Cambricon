import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
import os            # os包集成了一些对文件路径和目录进行操作的类
import matplotlib.pyplot as plt
import time


# 读取数据
data_dir = 'E:\\CoderLife\\demo\\DogsVSCats'

data_transform = {x:transforms.Compose([transforms.Resize([128,128]),
                                       transforms.ToTensor()]) for x in ['train', 'val']}   # 这一步类似预处理
image_datasets = {x:datasets.ImageFolder(root = os.path.join(data_dir,x),
                                        transform = data_transform[x]) for x in ['train', 'val']}  # 这一步相当于读取数据
dataloader = {x:torch.utils.data.DataLoader(dataset = image_datasets[x],
                                           batch_size = 4,
                                           shuffle = True) for x in ['train', 'val']}  # 读取完数据后，对数据进行装载

# 模型搭建
class Models(torch.nn.Module):
    def __init__(self):
        super(Models, self).__init__()
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
            torch.nn.Linear(32 * 32 * 256, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(512, 2))

    def forward(self, inputs):
        x = self.Conv(inputs)
        x = x.view(-1, 32 * 32 * 256)
        x = self.Classes(x)
        return x


model = Models()
print(model)
'''
# 保存和加载整个模型
torch.save(model, 'model.pth')
model_1 = torch.load('model.pth')
print(model_1)

# 仅保存和加载模型参数
torch.save(model.state_dict(), 'params.pth')
dic = torch.load('params.pth')
model.load_state_dict(dic)
print(dic)
'''
loss_f = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

Use_gpu = torch.cuda.is_available()
if Use_gpu:
    model = model.cuda()

epoch_n = 15
time_open = time.time()

for epoch in range(epoch_n):
    print('epoch {}/{}'.format(epoch, epoch_n - 1))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            # # 设置为True，会进行Dropout并使用batch mean和batch var
            print('training...')
            model.train(True)
        else:
            # # 设置为False，不会进行Dropout并使用running mean和running var
            print('validing...')
            model.train(False)

        running_loss = 0.0
        running_corrects = 0.0
        # 输出标号 和对应图片，下标从1开始
        for batch, data in enumerate(dataloader[phase], 1):
            X, Y = data
            # 将数据放在GPU上训练
            X, Y = Variable(X).cuda(), Variable(Y).cuda()
            # 模型预测概率
            y_pred = model(X)
            # pred，概率较大值对应的索引值，可看做预测结果，1表示行
            _, pred = torch.max(y_pred.data, 1)
            # 梯度归零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_f(y_pred, Y)
            # 训练 需要反向传播及梯度更新
            if phase == 'train':
                # 反向传播出现问题
                loss.backward()
                optimizer.step()
            # 损失和
            running_loss += loss.data.item()
            # 预测正确的图片个数
            running_corrects += torch.sum(pred == Y.data)
            # 训练时，每500个batch输出一次，训练loss和acc
            if batch % 500 == 0 and phase == 'train':
                print('batch{},trainLoss:{:.4f},trainAcc:{:.4f}'.format(batch, running_loss / batch,
                                                                        100 * running_corrects / (4 * batch)))
        # 输出每个epoch的loss和acc
        epoch_loss = running_loss * 4 / len(image_datasets[phase])
        epoch_acc = 100 * running_corrects / len(image_datasets[phase])
        print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))
time_end = time.time() - time_open
# 保存模型
torch.save(model, 'E:\\JetBrains\\PyCharm\\Programms\\computer vision\\modles\\Sep1_23\\02-128.pt')

# 加载模型
"""
dic = torch.load('params.pth')
model.load_state_dict(dic)
print(dic)
"""
print(time_end)






