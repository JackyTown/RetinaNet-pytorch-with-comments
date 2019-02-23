import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw


print('Loading model..')
net = RetinaNet()
net.load_state_dict(torch.load('./checkpoint/params.pth'))
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
img = Image.open('./image/000001.jpg')
w = h = 600
img = img.resize((w,h)) # 图片resize

print('Predicting..')
x = transform(img) # ToTensor & Normalize
x = x.unsqueeze(0)
x = Variable(x, volatile=True)
loc_preds, cls_preds = net(x) # 从网络中获取预测结果

print('Decoding..')
encoder = DataEncoder()
boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w,h))
# squeeze(): 去掉维数为1的维度
# boxes: (tensor) decode box locations, sized [#obj,4].
# labels: (tensor) class labels for each box, sized [#obj,].

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
