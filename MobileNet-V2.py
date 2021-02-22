from torch import nn
from torchsummary import summary

#####MobielNet V2
### Ref: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf

## With some points from PyTorch Official implementation of MobileNet-V2 : https://github.com/pytorch/vision/blob/62e185c7ee6df9938809bc0aec353bbcad1f1223/torchvision/models/mobilenetv2.py#L31



class Inverted_residual_Block(nn.Module):
    def __init__(self,expansion_rate,input_channels,output_channels,stride):
        super(Inverted_residual_Block,self).__init__()
        self.expansion_rate = expansion_rate
        self.stride = stride
        self.input_conv = nn.Conv2d(kernel_size=1,
                                    stride=stride,
                                    in_channels=input_channels,
                                    out_channels=( input_channels * self.expansion_rate))
        self.bn = nn.BatchNorm2d(( input_channels * self.expansion_rate))

        self.dw_conv = nn.Conv2d(kernel_size=3,
                                 groups= (input_channels * self.expansion_rate),
                                 in_channels=(input_channels * self.expansion_rate),
                                 stride=1,
                                 padding=1,
                                 out_channels= (input_channels * self.expansion_rate) )

        self.relu = nn.ReLU(inplace=True)
        self.pw_conv = nn.Conv2d(kernel_size=1,
                                 in_channels= (input_channels * self.expansion_rate),
                                 out_channels=output_channels)

    def forward(self,x):
        input = x
        x = self.input_conv(input)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dw_conv(x)
        x= self.bn(x)
        x = self.relu(x)
        x= self.pw_conv(x)
        return x



class MobileNetV2(nn.Module):

    def __init__(self,num_classes=1000):
        super(MobileNetV2,self).__init__()

        self.configuration  = [
            #(expansion_rate,n(number of blocks in this module),stride,out_channels)
            (1, 1, 1, 16),
            (6, 2, 2, 24),
            (6, 3, 2, 32),
            (6, 4, 2, 64),
            (6, 3, 1, 96),
            (6, 3, 2, 160),
            (6, 1, 1, 320),
        ]

        self.conv_1 = nn.Conv2d(in_channels=3,out_channels=32,stride=2,padding=1,kernel_size=3,bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layers = self._make_layers()
        self.conv_2 = nn.Conv2d(kernel_size=1,in_channels=320,out_channels=1280,bias=False)
        self.bn_2 = nn.BatchNorm2d(1280)
        self.avgPool = nn.AdaptiveAvgPool2d((1,1))


        #In the original paper, the last layer is a CNN with 1*1 sizes and 1000( number of classes) channels.
        #There is no fully-connted layer in the orginal paper. However In
        #PyTorch Implementation, the last CN has been replaced with a full_connected layer.

        #self.conv_3 = nn.Conv2d(kernel_size=1,in_channels=1280,out_channels=num_classes,bias=False)
        #self.bn_3 = nn.BatchNorm2d(num_classes)
        self.fc = nn.Linear(in_features=1280,out_features=num_classes)

    def forward(self,x):
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layers(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.avgPool(x)
        #x = self.conv_3(x)
        #x = self.bn_3(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(0.2)(x)
        x = self.fc(x)
        return x


    def _make_layers(self):

        layers= []

        for (expansion_rate,n,stride,out_channels) in self.configuration:
            #in each row of Table 2, page 5 of the paper.( Each module of Inverted residual_blocks)
            for number_of_blocks in range(n):
                if len(layers) == 0:
                    in_channels = 32
                if number_of_blocks == 0:
                    block_stride = stride
                else:
                    block_stride = 1

                layers.append(Inverted_residual_Block(expansion_rate=expansion_rate,
                                               input_channels=in_channels,
                                               output_channels=out_channels,
                                               stride=block_stride))


                in_channels = out_channels
        return nn.Sequential(*layers)






mobile_net = MobileNetV2()
summary(mobile_net,(3,224,224),device="cpu")



