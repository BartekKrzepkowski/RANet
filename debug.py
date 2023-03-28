import torch

from src.modules.methods.ran import RANet

import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()

args.block_step = 2
args.stepmode = 'even'
args.use_valid = False
args.out_channels = 16


args.nBlocks = 2
args.Block_base = 2 # Block_base vs block
args.step = 4
args.stepmode ='even'
args.compress_factor = 0.25
args.nChannels = 64 # n = number? stała liczba kanałów we wszystkich warstwach?
args.data = 'cifar10'
args.growthRate = 16 # growth rate of what exactly?

args.reduction = 0.5 # reduction of what?

args.bottlenectFactor = '4-2-2-1' # bn = bottleneck?
args.scale_list = '1-2-3-4'
args.growFactor = '4-2-2-1' # gr = growth?


args.bottlenectFactor = list(map(int, args.bottlenectFactor.split('-')))
args.scale_list = list(map(int, args.scale_list.split('-')))
args.growFactor = list(map(int, args.growFactor.split('-')))
args.nScales = len(args.growFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

args.num_classes = 10




x_true = torch.randn(16, 3, 32, 32)

model = RANet(args)

output = model(x_true)
print(len(output))

# describe the model using torch modules
# print(model)


