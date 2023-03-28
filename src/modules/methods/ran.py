import math
import copy
from typing import List

import torch
import torch.nn.functional as F

from src.modules.architectures.models import ConvBasic, ConvWithBottleneck
from src.utils import common


class ResAdapInitial(torch.nn.Module):
    def __init__(self, scale_list, _grFactor, down_scale_params):
        super().__init__()
        in_channels = down_scale_params['in_channels']
        out_channels = down_scale_params['out_channels']
        del down_scale_params['in_channels']
        del down_scale_params['out_channels']
        self.layers = torch.nn.ModuleList()
        self.layers.append(ConvBasic(in_channels=in_channels,
                                     out_channels=out_channels*_grFactor[0],
                                     stride=1,
                                     **down_scale_params))

        for i in range(1, len(scale_list)):
            in_channels = out_channels * _grFactor[i-1]
            stride = 1 if scale_list[i-1] == scale_list[i] else 2
            self.layers.append(ConvBasic(in_channels=in_channels,
                                         out_channels=out_channels*_grFactor[i],
                                         stride=stride,
                                         **down_scale_params))

    def forward(self, x):
        # res[0] with the smallest resolutions
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res[::-1]
    

class ClassifierModule(torch.nn.Module):
    def __init__(self, pre_cls, channels, num_classes):
        super().__init__()
        self.pre_cls = pre_cls
        self.linear = torch.nn.Linear(channels, num_classes)

    def forward(self, x):
        x = self.pre_cls(x)
        x = x.view(x.size(0), -1)
        return self.linear(x)
    

class ConvNormal(torch.nn.Module):
    def __init__(self, in_channels, out_channels, isBatchNormAfter, bottleneckFactor):
        '''
        The convolution with normal connection.
        '''
        super().__init__()
        self.conv_normal = ConvWithBottleneck(in_channels, out_channels, whether_down_sample=False,
                                   isBatchNormAfter=isBatchNormAfter, bottleneckFactor=bottleneckFactor)

    def forward(self, x):
        if not isinstance(x, list): #kiedy to jest listą???
            x = [x]
        results = [x[0], self.conv_normal(x[0])]
        results = torch.cat(results, dim=1)
        return results
    

class ConvUpNormal(torch.nn.Module):
    def __init__(self, in_channels1, in_channels2, out_channels, isBatchNormAfter,
                 bottleneckFactor1, bottleneckFactor2, compress_factor, whether_down_sample):
        '''
        The convolution with normal and up-sampling connection.
        '''
        super().__init__()
        self.conv_normal = ConvWithBottleneck(in_channels=in_channels1, out_channels=out_channels-math.floor(out_channels*compress_factor),
                                              whether_down_sample=whether_down_sample, isBatchNormAfter=isBatchNormAfter, bottleneckFactor=bottleneckFactor1)
        self.conv_up = ConvWithBottleneck(in_channels=in_channels2, out_channels=math.floor(out_channels*compress_factor),
                                          whether_down_sample=False, isBatchNormAfter=isBatchNormAfter, bottleneckFactor=bottleneckFactor2)
        

    def forward(self, x):
        # x[0] is the low resolution feature map
        results = self.conv_normal(x[1])
        _,_,h,w = results.size() # możliwy downsample na h i w
        results = [F.interpolate(x[1], size=(h,w), mode = 'bilinear', align_corners=True), # gdy nie ma downsample to nie ma interpolacji
               F.interpolate(self.conv_up(x[0]), size=(h,w), mode = 'bilinear', align_corners=True),
               results]
        results = torch.cat(results, dim=1)
        return results
    

class _BlockNormal(torch.nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, reduction_rate, trans, bottleneckFactor):
        '''
        The basic computational block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        '''
        super().__init__()
        self.trans_flag = trans
        self.num_layers = num_layers
        conv_normal_boiler_params = dict(out_channels=growth_rate, isBatchNormAfter=True, bottleneckFactor=bottleneckFactor)
        self.layers = torch.nn.ModuleList([ConvNormal(in_channels + i * growth_rate, **conv_normal_boiler_params)
                                           for i in range(num_layers)])
        
        if self.trans_flag:
            out_channels = in_channels + num_layers * growth_rate
            self.trans = ConvBasic(out_channels, math.floor(1.0 * reduction_rate * out_channels),
                                   kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        output = [x]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            output.append(x)
        if self.trans_flag:
            x = self.trans(x)
        return x, output
    
    def _blockType(self):
        return 'normal'
    

class _BlockUpNormal(torch.nn.Module):
    def __init__(self, num_layers, in_channels, nIn_lowFtrs, growth_rate,
                reduction_rate, trans, whether_down_sample, compress_factor, bottleneckFactor1, bottleneckFactor2):
        '''
        The basic fusion block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        compress_factor: There will be compress_factor*100% information from the previous
                sub-network.  
        '''
        super().__init__()
        self.trans_flag = trans
        self.layers = torch.nn.ModuleList()
        self.num_layers = num_layers
        conv_upnormal_boiler_params = dict(out_channels=growth_rate, isBatchNormAfter=True, compress_factor=compress_factor,
                                           bottleneckFactor1=bottleneckFactor1, bottleneckFactor2=bottleneckFactor2)
        for i in range(num_layers-1):
            self.layers.append(ConvUpNormal(in_channels1=in_channels+i*growth_rate, in_channels2=nIn_lowFtrs[i], whether_down_sample=False, **conv_upnormal_boiler_params))

        self.layers.append(ConvUpNormal(in_channels1=in_channels+(i+1)*growth_rate, in_channels2=nIn_lowFtrs[i+1], whether_down_sample=whether_down_sample, **conv_upnormal_boiler_params))
        
        out_channels = in_channels + num_layers * growth_rate
        self.conv_last = ConvBasic(nIn_lowFtrs[num_layers], math.floor(out_channels*compress_factor), # dlaczego nie jest to częścią conv_up_normal?
                                   kernel_size=1, stride=1, padding=0)
        if self.trans_flag:
            out_channels = out_channels + math.floor(out_channels * compress_factor)
            self.trans = ConvBasic(out_channels, math.floor(1.0*reduction_rate*out_channels),
                                   kernel_size=1, stride=1, padding=0)
            
    def forward(self, x, low_feat):
        output = [x]
        for i in range(self.num_layers):
            inp = [low_feat[i]]
            inp.append(x)
            x = self.layers[i](inp)
            output.append(x)
        _, _, h, w = x.size()
        x = [x]
        x.append(F.interpolate(self.conv_last(low_feat[self.num_layers]),
                               size=(h,w), mode = 'bilinear', align_corners=True))
        x = torch.cat(x, dim=1)
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'up'
    

class RANet(torch.nn.Module):
    def __init__(self, args, criterion):
        super().__init__()
        self.criterion = criterion
        self.scale_flows = torch.nn.ModuleList() #????
        self.classifier = torch.nn.ModuleList()
        
        self.compress_factor = args.compress_factor #?
        self.bottlenectFactor = copy.copy(args.bottlenectFactor)

        self.nScales = len(args.scale_list) # 4

        # The number of blocks in each scale flow
        self.nBlocks = [0] #?
        for i in range(self.nScales):
            self.nBlocks.append(args.block_step * i + args.nBlocks) # [0, 2, 4, 6, 8]
        
        # The number of layers in each block
        # self.steps = args.step

        down_scale_params = {
            'in_channels': 3,
            'out_channels': args.out_channels,
            'kernel_size': 3,
            'padding': 1,
        }
        self.initial_layer = ResAdapInitial(args.scale_list[::-1], args.growFactor[::-1], down_scale_params)

        self.create_model(args)

        for block_module in self.scale_flows:
            for _m in block_module.modules():
                self._init_weights(_m)

        for block_module in self.classifier:
            for _m in block_module.modules():
                self._init_weights(_m)

    def create_model(self, args):
        scale_list = args.scale_list
        num_layers_given_block = [args.step]
        for ii in range(self.nScales):

            scale_flow = torch.nn.ModuleList()

            n_current_block = 1
            in_channels = args.out_channels * args.growFactor[ii] # growFactor = [4,2,2,1]
            growth_rate = args.growthRate * args.growFactor[ii] # 16 * [4,2,2,1]
            _nIn_lowFtrs = []
            
            for i in range(self.nBlocks[ii+1]):
                # If transiation ???????????????
                trans = self._trans_flag(n_current_block, n_block_all = self.nBlocks[ii+1], inScale = scale_list[ii])

                if n_current_block > self.nBlocks[ii]: # block który zwyczajnie przetwarza dane
                    block_module, block_out_channels = self._build_normal_block(in_channels, num_layers_given_block[n_current_block-1], growth_rate,
                                                                                args.reduction, trans, bottlenectFactor=self.bottlenectFactor[ii])
                    if args.stepmode == 'even':
                        num_layers_given_block.append(args.step)
                    elif args.stepmode == 'lg':
                        num_layers_given_block.append(args.step + num_layers_given_block[-1])
                    else:
                        raise NotImplementedError
                else: # block który integruje wyjścia z niższych skal
                    down = n_current_block in self.nBlocks[:ii+1][-(scale_list[ii]-1):] # czy jest to wyliczone czy zahardkodowane?
                    block_module, block_out_channels = self._build_upNormal_block(in_channels, nIn_lowFtrs[i], num_layers_given_block[n_current_block-1], growth_rate,
                                                                                  args.reduction, trans, down=down, bottlenectFactor1=self.bottlenectFactor[ii], bottlenectFactor2=self.bottlenectFactor[ii-1])

                in_channels = block_out_channels[-1]
                _nIn_lowFtrs.append(block_out_channels)
                scale_flow.append(block_module)
                
                if n_current_block > self.nBlocks[ii]:
                    self.classifier.append(self._build_classifier_cifar(in_channels, 10))
                    
                n_current_block += 1
                
            nIn_lowFtrs = _nIn_lowFtrs
            self.scale_flows.append(scale_flow)

        args.num_exits = len(self.classifier)

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.bias.data.zero_()

    def _build_normal_block(self, in_channels, num_layers, growth_rate, reduction_rate, trans, bottlenectFactor=2):
        '''Build a normal block
        # Arguments
            in_channels: int, the number of input channels
            step: int, the number of layers in the block
            growth_rate: int, 
            reduction_rate: int,
            
        # Returns
            block: torch.nn.Module, the block
            out_channels: int, the number of output channels
            '''
        block = _BlockNormal(num_layers, in_channels, growth_rate, reduction_rate,
                             trans, bottleneckFactor=bottlenectFactor)
        out_channels = [in_channels + i * growth_rate for i in range(num_layers+1)]
      
        if trans:
            out_channel = math.floor(1.0 * reduction_rate * out_channel)
            out_channels.append(out_channel) #???

        return block, out_channels

    def _build_upNormal_block(self, in_channels, nIn_lowFtr, num_layers, growth_rate, reduction_rate, trans, down, bottlenectFactor1=1, bottlenectFactor2=2):       
        compress_factor = self.compress_factor

        block = _BlockUpNormal(num_layers, in_channels, nIn_lowFtr, growth_rate, reduction_rate,
                               trans, down, compress_factor, bottleneckFactor1=bottlenectFactor1, bottleneckFactor2=bottlenectFactor2)
        out_channels = [in_channels + i * growth_rate for i in range(num_layers+1)]
        out_channel = out_channels[-1] + math.floor(out_channels[-1]*compress_factor)
    
        if trans:
            out_channel = math.floor(1.0 * reduction_rate * out_channel)

        out_channels.append(out_channel)

        return block, out_channels

    def _trans_flag(self, n_block_curr, n_block_all, inScale): # można efektywniej?
        flag = False
        for i in range(inScale-1):
            if n_block_curr == math.floor((i + 1) * n_block_all / inScale):
                flag = True
        return flag

    def forward(self, x):
        inp = self.initial_layer(x)
        outputs, low_ftrs = [], []
        classifier_idx = 0
        for ii in range(self.nScales):
            _x = inp[ii]
            _low_ftrs = []
            n_block_curr = 0
            for i in range(self.nBlocks[ii+1]):
                if self.scale_flows[ii][i]._blockType() == 'normal':
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                n_block_curr += 1
                
                if n_block_curr > self.nBlocks[ii]:
                    y = self.classifier[classifier_idx](_x)
                    outputs.append(y)
                    classifier_idx += 1
                
            low_ftrs = _low_ftrs        
        return outputs

    def _build_classifier_cifar(self, in_channels, num_classes):
        interChannels1, interChannels2 = 128, 128
        conv = torch.nn.Sequential(
            ConvBasic(in_channels, interChannels1, kernel_size=3, stride=2, padding=1),
            ConvBasic(interChannels1, interChannels2, kernel_size=3, stride=2, padding=1),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = torch.nn.Sequential(
            ConvBasic(nIn, nIn, kernel_size=3, stride=2, padding=1),
            ConvBasic(nIn, nIn, kernel_size=3, stride=2, padding=1),
            torch.nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)
    
    def calc_loss(self, y_preds, y_true):
        loss = 0.0
        evaluators = {}
        for i, y_pred in enumerate(y_preds):
            loss_, evaluators_ = self.criterion(y_pred, y_true, postfix=f'_{i}')
            loss += loss_
            evaluators.update(evaluators_)
        evaluators['overall_loss'] = loss.item()
        return loss, evaluators