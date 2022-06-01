import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Conv2DTranspose
from paddle.fluid.dygraph.base import to_variable

class DoubleCovn(fluid.dygraph.Layer):
    def __init__(self, inchannel, outchannel):
        super(DoubleCovn, self).__init__()
        self.layers = fluid.dygraph.Sequential(
            Conv2D(inchannel, outchannel, filter_size=3,stride=1, padding=1),
            fluid.BatchNorm(outchannel,act='relu'),
            Conv2D(outchannel, outchannel, filter_size=3, stride=1,padding=1),
            fluid.BatchNorm(outchannel, act='relu'),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Unet(fluid.dygraph.Layer):
    def __init__(self, input, out):
        super(Unet, self).__init__()

        self.c1 = DoubleCovn(input, 64)
        self.maxpool1 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c2 = DoubleCovn(64, 128)
        self.maxpool2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c3 = DoubleCovn(128, 256)
        self.maxpool3 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c4 = DoubleCovn(256, 512)
        self.maxpool4 = Pool2D(pool_size=2,pool_type='max',pool_stride=2,)
        self.c5 = DoubleCovn(512, 1024)

        self.up6 = Conv2DTranspose(1024, 512, 2, stride=2)
        self.c6 = DoubleCovn(1024, 512)
        self.up7 = Conv2DTranspose(512, 256, 2, stride=2)
        self.c7 = DoubleCovn(512, 256)
        self.up8 = Conv2DTranspose(256, 128, 2, stride=2)
        self.c8 = DoubleCovn(256, 128)
        self.up9 = Conv2DTranspose(128, 64, 2, stride=2)
        self.c9 = DoubleCovn( 128, 64)

        self.c10 = Conv2D(64, out, 1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.maxpool1(c1)
        c2 = self.c2(p1)
        p2 = self.maxpool2(c2)
        c3 = self.c3(p2)
        p4 = self.maxpool3(c3)
        c4 = self.c4(p4)
        p5 = self.maxpool4(c4)
        c5 = self.c5(p5)

        up6 = self.up6(c5)
        merge6 = fluid.layers.concat([up6, c4], axis=1)
        c6 = self.c6(merge6)
        up7 = self.up7(c6)
        merge7 = fluid.layers.concat([up7, c3], axis=1)
        c7= self.c7(merge7)
        up8 = self.up8(c7)
        merge8 = fluid.layers.concat([up8, c2], axis=1)
        c8 = self.c8(merge8)
        up9 = self.up9(c8)
        merge9 = fluid.layers.concat([up9 , c1], axis=1)
        c9 = self.c9(merge9)

        c10 = self.c10(c9)
        out = fluid.layers.logsigmoid(c10)
        return out        