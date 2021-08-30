import argparse
import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader
from model import Model
from dataset import Dataset
import os
from tqdm import tqdm
from predict import predict

def train(dataset, model, args):
    model.train()


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    lossAct = []
    initEpoch = 0
    if args.checkPoint:
        loadDict = torch.load(os.path.join('Checkpoints/Checkpoint99.pth'))
        model.load_state_dict(loadDict['Model'])
        optimizer.load_state_dict(loadDict['Optimizer'])
        lossAct = loadDict['LossAct']
        initEpoch = loadDict['Epoch']

    for epoch in range(initEpoch,args.max_epochs):
        state_h, state_c = model.init_state(args.sequence_length)
        totLoss = 0
        for batch, (x, y) in enumerate(tqdm(dataloader)):
            #breakpoint()
            optimizer.zero_grad()

            y_pred, (state_h, state_c) = model(x, (state_h, state_c))
            loss = criterion(y_pred.transpose(1, 2), y)
            totLoss += loss.item()
            
            state_h = state_h.detach()
            state_c = state_c.detach()

            loss.backward()
            optimizer.step()

            # print({ 'epoch': epoch, 'batch': batch, 'loss': loss.item() })
        stateDict = model.state_dict()
        optAct = optimizer.state_dict()
        lossAct.append(totLoss/(batch+1))
        saveDict = {'Epoch': epoch+1,'Model':stateDict,'Optimizer':optAct,'LossAct': lossAct}
        torch.save(saveDict, os.path.join(f'Checkpoints/Checkpoint{epoch}.pth'))
        print({ 'epoch': epoch, 'loss': lossAct[-1]})

parser = argparse.ArgumentParser()
parser.add_argument('--checkPoint', action='store_true')
parser.add_argument('--max-epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--sequence-length', type=int, default=10)
args = parser.parse_args()

dataset = Dataset(args)
model = Model(dataset)

train(dataset, model, args)
predict(dataset, model, text='Knock knock. Whos there?')
