import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import tqdm


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.__build__()

    def __build__(self):
        filter_size = 32
        self.conv1 = nn.Conv2d(3, filter_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filter_size, filter_size*2 , kernel_size=3, stride=1, padding=1)

        filter_size = filter_size*2
        self.conv3 = nn.Conv2d(filter_size*30, filter_size, kernel_size=3, stride=1, padding=1)
        self.batch1 = nn.BatchNorm2d(filter_size)

        self.conv4 = nn.Conv2d(filter_size, filter_size*2, kernel_size=3, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(filter_size*2)

        filter_size = filter_size*2
        self.lin1 = nn.Linear(filter_size, 128)
        self.drop = nn.Dropout()
        self.output = nn.Linear(128, 21)
        self.max_pool = nn.MaxPool2d((2,2))

    def forward(self, x):
        batch_size, frames, c, h, w = x.shape

        inter_conv = []
        for i in range(frames):
            inter_input = torch.reshape(x[:, i, :, :, :], (batch_size, c, h, w))
            out = self.max_pool(F.selu(self.conv1(inter_input)))
            out = F.selu(self.conv2(out))
            inter_conv.append(out)

        inter_conv = torch.cat(inter_conv, axis=1)
        conv = self.max_pool(F.selu(self.batch1(self.conv3(inter_conv))))
        conv = self.max_pool(F.selu(self.batch2(self.conv4(conv))))
        g_pool = conv.mean(axis=(2,3))
        lin = F.selu(self.lin1(g_pool))
        #lin = F.selu(self.lin1(g_pool))
        out = self.output(lin)
        return out

class TextModel(torch.nn.Module):
    def __init__(self, count):
        super(TextModel, self).__init__()
        self.count = count
        self.__build__()

    def __build__(self):
        self.lin1 = nn.Linear(self.count, 64)
        self.lin2 = nn.Linear(64, 128)
    
    def forward(self, x):
        x = F.selu(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        print('text', x.shape)
        return x

class ModelMerge(nn.Module):
    def __init__(self, main_model, text_model):
        super(ModelMerge, self).__init__()
        self.mm = main_model
        self.text_model = text_model
    
    def forward(self, img, text):
        text_result = self.text_model(text)

        print(img.shape)

        batch_size, frames, c, h, w = img.shape

        inter_conv = []
        for i in range(frames):
            inter_input = torch.reshape(img[:, i, :, :, :], (batch_size, c, h, w))
            out = self.mm.max_pool(F.selu(self.mm.conv1(inter_input)))
            out = F.selu(self.mm.conv2(out))
            inter_conv.append(out)

        inter_conv = torch.cat(inter_conv, axis=1)
        conv = self.mm.max_pool(F.selu(self.mm.batch1(self.mm.conv3(inter_conv))))
        conv = self.mm.max_pool(F.selu(self.mm.batch2(self.mm.conv4(conv))))
        g_pool = conv.mean(axis=(2,3))
        lin = F.selu(self.mm.lin1(g_pool))
        print('img', lin.shape)
        lin = lin * text_result
        #lin = F.selu(self.lin1(g_pool))
        out = self.mm.output(lin)
        return out
        
        
 


def label_from_path(path):
    name = path.split('/')[-1].split('.')[0]
    return '{}'.format(name)
    # return name

def count_load_video(path):
    count  = 0
    vc = cv2.VideoCapture(path)
    images = []
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        h,w,c = frame.shape
        length = min(h//3, w//3)
        frame = frame[h//2-length//2:h//2+length//2, w//2-length//2:w//2+length//2, :]
        frame = cv2.resize(frame, (256,256))
        last_frame = frame
        if count % 2 == 0:
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        count = count + 1
    if len(images) < 30:
        while(len(images)<30):
            images.append(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
    return images, label_from_path(path)


def make_label_list():
    labels = []
    for i in range(1,22):
        file = open('./align/{}.txt'.format(i), 'r', encoding='utf-8')
        labels.append(file.readline().strip())
    return labels


if __name__ == "__main__":
    labels = make_label_list()
    print(labels)
    root_dir = './data'
    data_path = [os.path.join(root_dir, name) for name in os.listdir(root_dir)]

    video_pathes = []

    md = Model()
    tx = TextModel(50)
    merge = ModelMerge(md, tx)
    merge.cuda()
    
    img = torch.rand((1,30,3,256,256)).type(torch.FloatTensor).cuda()
    text = torch.rand((1, 50)).type(torch.FloatTensor).cuda()

    result = merge(img, text)
    
    
    for dir_path in data_path:
        video_path = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        for ph in video_path:
            video_pathes.append(ph)
    print(video_pathes)
    indexes = np.arange(len(video_path))
    batch_size = 8
    epoches = 1000
    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(md.parameters(), lr=5e-4)
    max_acc = 0
    for epoch in range(1, epoches+1):
        random.shuffle(indexes)
        loss = 0
        acc = 0
        predicted = []
        target = []

        md.train()
        for iter in tqdm.tqdm(range(len(indexes)//batch_size)):
            x = []
            y = []
            optim.zero_grad()
            for batch_idx in range(batch_size):
                b_idx = batch_size*iter + batch_idx
                path = video_path[indexes[b_idx]]
                frames, label = count_load_video(path)

                h, w, c = frames[0].shape

                frames = np.asarray(frames[:30], dtype=np.float)
                frames = np.transpose(frames, (0, 3, 1, 2))
                #print(frames.shape)
                frames = np.reshape(frames, (30, c, h, w))

                # frames = torch.from_numpy(frames).type(torch.FloatTensor)
                label = labels.index(label)
                x.append(frames)
                y.append(label)
            x = np.asarray(x, dtype=np.float)/255
            x = torch.from_numpy(x).type(torch.FloatTensor)
            y = torch.from_numpy((np.array(y))).type(torch.LongTensor)
            x = x.cuda()
            y = y.cuda()
            out = md(x)
            result = criterion(out, y)
            result.backward()
            optim.step()
            loss = loss + result.item()
            r_np = out.detach().cpu().numpy()
            r_idx = np.argmax(r_np, axis=1)
            del out

        loss = loss/((len(indexes)//batch_size))

        md.eval()
        for iter in tqdm.tqdm(range(len(indexes)//batch_size)):
            x = []
            y = []
            for batch_idx in range(batch_size):
                b_idx = batch_size*iter + batch_idx
                path = video_path[indexes[b_idx]]
                frames, label = count_load_video(path)

                h, w, c = frames[0].shape

                frames = np.asarray(frames[:30], dtype=np.float)
                frames = np.transpose(frames, (0, 3, 1, 2))
                #print(frames.shape)
                frames = np.reshape(frames, (30, c, h, w))

                # frames = torch.from_numpy(frames).type(torch.FloatTensor)
                label = labels.index(label)
                x.append(frames)
                y.append(label)
            x = np.asarray(x, dtype=np.float)/255
            x = torch.from_numpy(x).type(torch.FloatTensor)
            y = torch.from_numpy((np.array(y))).type(torch.LongTensor)
            x = x.cuda()
            y = y.cuda()
            out = md(x)
            r_np = out.detach().cpu().numpy()
            r_idx = np.argmax(r_np, axis=1)
            acc = acc + np.sum(r_idx == y.detach().cpu().numpy())
            predicted.append(list(r_idx))
            target.append(list(y.detach().cpu().numpy()))
            del out

        acc = acc/((len(indexes)//batch_size * batch_size))

        if acc > max_acc:
            max_acc = acc
            torch.save(md.state_dict(), 'staet_dict.dict')
            traced = torch.jit.trace(md, torch.rand(1,30,3,256,256).type(torch.FloatTensor).cuda())
#            traced.save(md.state_dict(), 'traced.pt')
            print("Saved best model")
        if acc*100>60:
            exit(0)
            break
        print(epoch, loss, acc*100)
        print(predicted)
        print(target)

