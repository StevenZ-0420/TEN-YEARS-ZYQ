import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

writer = SummaryWriter('./logs/')

#%% 训练设备
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% 参数定义
EPOCH = 10
BATCH_SIZE = 128
LR = 1E-3

#%% 下载数据集
train_file = datasets.MNIST(
    root='C:/Users/淦掉皮质醇/Desktop/MINST_Data',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
test_file = datasets.MNIST(
    root='C:/Users/淦掉皮质醇/Desktop/MINST_Data',
    train=False,
    transform=transforms.ToTensor()
)

#%% 数据可视化
train_data = train_file.data
train_targets = train_file.targets
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(train_targets[i].numpy())
    plt.axis('off')
    plt.imshow(train_data[i], cmap='gray')
plt.show()

test_data = test_file.data
test_targets = test_file.targets
plt.figure(figsize=(9, 9))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.title(test_targets[i].numpy())
    plt.axis('off')
    plt.imshow(test_data[i], cmap='gray')
plt.show()

#%% 制作数据加载器
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#%% 模型结构
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y

#%% 创建模型
model = CNN().to(device)
optim = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()

#%% 定义计算整个训练集或测试集loss及acc的函数
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            output = model(data)
            loss += lossf(output, targets).item()  # 修正损失计算
            correct += (output.argmax(1) == targets).sum()
            total += data.size(0)
    loss /= len(data_loader)  # 计算平均损失
    acc = correct.item() / total
    return loss, acc

#%% 训练过程打印函数
model_saved_list = []
temp = 0

def show(epoch):
    global model_saved_list, temp
    loss, acc = calc(train_loader)
    writer.add_scalar('loss', loss, epoch + 1)
    writer.add_scalar('acc', acc, epoch + 1)

    val_loss, val_acc = calc(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch + 1)
    writer.add_scalar('val_acc', val_acc, epoch + 1)

    if val_acc > temp:
        model_saved_list = [
            f'EPOCH: {epoch + 1}/{EPOCH}',
            f'LOSS: {loss:.4f}',
            f'ACC: {acc:.4f}',
            f'VAL-LOSS: {val_loss:.4f}',
            f'VAL-ACC: {val_acc:.4f}'
        ]
        torch.save(model.state_dict(), 'model.pt')
        temp = val_acc

#%% 训练模型
for epoch in range(EPOCH):
    start_time = time.time()
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data.to(device)
        targets = targets.to(device)
        output = model(data)
        loss = lossf(output, targets)
        acc = (output.argmax(1) == targets).sum().item() / data.size(0)  # 修正准确率计算
        loss.backward()
        optim.step()
        print(
            f'EPOCH: {epoch + 1}/{EPOCH}',
            f'STEP: {step + 1}/{len(train_loader)}',
            f'LOSS: {loss.item():.4f}',
            f'ACC: {acc:.4f}',
            end='\r'
        )
    show(epoch)  # 在每个epoch结束后调用show
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time - start_time)}')

#%% 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | ' + model_saved_show)
with open('model.txt', 'a') as f:
    f.write(model_saved_show + '\n')



# 定义加载和识别手写数字的函数
def recognize_handwritten_digit(image_path):
    # 加载并预处理图片
    image = Image.open(image_path).convert('L')  # 转为灰度图
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # 调整大小为28x28
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 进行归一化
    ])
    image = transform(image).unsqueeze(0)  # 增加 batch 维度

    # 使用模型进行预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        output = model(image.to(device))
        pred = output.argmax(dim=1, keepdim=True)  # 找到最大值的索引
        print(f"识别结果: {pred.item()}")  # 打印识别结果

# 使用手写数字图片进行识别
recognize_handwritten_digit('C:/Users/淦掉皮质醇/Desktop/MINST_Data/img_zyq.jpg')