import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
import tqdm

output_filename = './mergeOut.txt'
if os.path.isfile(output_filename):
	os.remove(output_filename)
f_out = open(output_filename,'a')

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
        self.lin1 = nn.Linear(filter_size, filter_size)
        self.drop = nn.Dropout()
        self.output = nn.Linear(filter_size, 21)
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
        #lin = self.drop(F.selu(self.lin1(g_pool)))
        lin = F.selu(self.lin1(g_pool))
        out = self.output(lin)
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
    batch_size =1
    labels = make_label_list()
    print(labels)
    root_dir = './data'
    data_path = [os.path.join(root_dir, name) for name in os.listdir(root_dir)]

    video_pathes = []

    for dir_path in data_path:
        video_path = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
        for ph in video_path:
            video_pathes.append(ph)

    indexes = np.arange(len(video_path))

    #md = torch.jit.load('traced2.pt')
    md = Model()
    #md.load_state_dict(torch.load('staet_dict.dict'))
    md.load_state_dict(torch.load('best.dict')) 
    md.cuda()
    md.eval()

    for epoch in range(1, 1+1):
        predicted = []
        target = []
        acc=0
        for iter in tqdm.tqdm(range(len(indexes)//1)):
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
                if label in labels:
                    label = labels.index(label)
                else:
                    label = 0
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

        acc = acc/(len(indexes))

        predicted = [y for x in predicted for y in x]
        print([labels[p] for p in predicted])
        f_out.write(labels[predicted[0]])
        target = [y for x in target for y in x]
        print([labels[t] for t in target])
        print('{}%'.format(int(acc*100)))

