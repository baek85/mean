import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    " 3x3 convolution with padding "
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)
def bnlayer(in_planes):
    return nn.BatchNorm2d(in_planes)
    #return MeanOnlyBatchNorm(in_planes)

class MeanOnlyBatchNorm(nn.Module):
    def __init__(self, num_features, momentum=0.1):
        super(MeanOnlyBatchNorm, self).__init__()
        self.num_features = num_features
        self.momentum = momentum
        self.weight = Parameter(torch.Tensor(num_features))
        self.bias = Parameter(torch.Tensor(num_features))

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, inp):
        size = list(inp.size())
        gamma = self.weight.view(1, self.num_features, 1, 1)
        beta = self.bias.view(1, self.num_features, 1, 1)

        if self.train:
            avg = torch.mean(inp.view(size[0], self.num_features, -1), dim=2)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * torch.mean(avg.data, dim=0)
        else:
            avg = Variable(self.running_mean.repeat(size[0], 1), requires_grad=False)

        output = inp - avg.view(size[0], size[1], 1, 1)
        output = output*gamma + beta

        return output

class ConvLarge(nn.Module):

    def __init__(self, args, num_classes=10):
        super(ConvLarge, self).__init__()
        self.args = args
        #self.fc = args.feature_channels
        self.leaky = 0.1
        self.fc = 128
        self.dropratio = 0.5
        self.noise = Variable(torch.zeros(1,3,32,32).cuda())
        self.std   = 0.15
        #self.noise = GaussianNoise(self.std)
        self.layer1 = self._make_layer([3, self.fc, self.fc, self.fc])
        self.drop1 = nn.Dropout2d(p = self.dropratio)
        self.layer2 = self._make_layer([self.fc, 2*self.fc, 2*self.fc, 2*self.fc])
        self.drop2 = nn.Dropout2d(p = self.dropratio)
        self.layer3 = nn.Sequential(
            weight_norm(nn.Conv2d(2*self.fc, 4*self.fc, kernel_size=3, stride=1, padding=0, bias=False)),
            bnlayer(4*self.fc),
            nn.LeakyReLU(self.leaky, inplace=True),
            weight_norm(nn.Conv2d(4*self.fc, 2*self.fc, kernel_size=1, stride=1, padding=0, bias=False)),
            bnlayer(2*self.fc),
            nn.LeakyReLU(self.leaky, inplace=True),
            weight_norm(nn.Conv2d(2*self.fc, self.fc, kernel_size=1, stride=1, padding=0, bias=False)),
            bnlayer(self.fc),
            nn.LeakyReLU(self.leaky, inplace=True)
        )
        self.avgpool = nn.AvgPool2d(6, stride=1)
        self.fclayer = nn.Linear(self.fc, num_classes)

    def _make_layer(self, channels, bn=True, act='lrelu'):

        layers = []
        for idx in range(len(channels)-1):
            layers.append(weight_norm(conv3x3(channels[idx], channels[idx+1])))
            if bn:
                layers.append(bnlayer(channels[idx+1]))

            if(act == 'lrelu'):
                layers.append(nn.LeakyReLU(self.leaky, inplace=True))
        layers.append(nn.MaxPool2d(2,2))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(0.0, self.std)
            x = self.noise + x


        x = self.layer1(x)
        x = self.drop1(x)

        x = self.layer2(x)
        x = self.drop2(x)

        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fclayer(x)

        return x

    def weight_init(self):
        for m in self.layer1:
            weights_init_kaiming(m)
        for m in self.layer2:
            weights_init_kaiming(m)
        for m in self.layer3:
            weights_init_kaiming(m)
        

def weights_init_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif class_name.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()




def baseline(args):
    model = ConvLarge(args)
    return model


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])
