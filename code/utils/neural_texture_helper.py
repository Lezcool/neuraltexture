import torch
import kornia
from models.vgg.vgg import vgg19


class VGGFeatures(torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)

        vgg_pretrained_features = vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(4):  # relu_1_1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):  # relu_2_1
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 18):  # relu_3_1
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(18, 27):  # relu_4_1
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(27, 36):  # relu_5_1
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):

        ## normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        device = 'cuda' if x.is_cuda else 'cpu'
        mean = torch.as_tensor(mean, device=device).view(1, 3, 1, 1)
        std = torch.as_tensor(std, device=device).view(1, 3, 1, 1)
        x = x.sub(mean)
        x = x.div(std)

        # get features
        h1 = self.slice1(x)
        h_relu1_1 = h1
        h2 = self.slice2(h1)
        h_relu2_1 = h2
        h3 = self.slice3(h2)
        h_relu3_1 = h3
        h4 = self.slice4(h3)
        h_relu4_1 = h4
        h5 = self.slice5(h4)
        h_relu5_1 = h5

        return [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]


class GramMatrix(torch.nn.Module):
    #计算输入张量的格拉姆矩阵（Gram Matrix）。
    def forward(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, h * w)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))

        gram_matrix.div_(h * w)
        return gram_matrix


def get_position(size, dim, device, batch_size):
    '''
    这个函数的作用是生成位置张量，用于表示图像上每个像素的位置信息。每次调用，value都会变。
    函数的执行过程如下：
    从 size 中获取图像的高度和宽度，并计算宽高比（aspect ratio）。
    使用 kornia.utils.create_meshgrid 函数生成一个网格矩阵，表示图像上每个像素的坐标位置。这个网格矩阵的形状是 (height, width, 2)，其中最后一个维度表示 x 和 y 坐标。
    对生成的网格矩阵进行一些处理。首先，将 y 坐标进行翻转，即乘以宽高比并取负值，以适应图像坐标系。然后，根据 dim 的值进行不同的处理。
    如果 dim 等于 1，表示生成一维位置张量，则只保留 x 坐标部分，将位置张量的形状变为 (batch_size, 1, height, width)。
    如果 dim 等于 3，表示生成三维位置张量，则需要在 x、y、z 坐标之间进行调换，并为 z 坐标赋予随机值，取值范围为 [-1, 1]。最后将 x、y、z 三个坐标合并，得到形状为 (batch_size, 3, height, width) 的位置张量。
    将生成的位置张量进行扩展，使其形状变为 (batch_size, dim, height, width)，其中 batch_size 是批量大小，dim 是位置张量的维度。

    这个函数的作用是为每个像素生成一个位置张量，该张量可以用于在图像处理或计算机视觉任务中，与其他特征进行融合或参考。位置张量可以提供图像上像素的空间信息，对于某些任务（如姿态估计、光流估计等）非常有用。
    return batch_size, dim, height, width
    '''
    height, width = size
    aspect_ratio = width / height
    position = kornia.utils.create_meshgrid(height, width, device=torch.device(device)).permute(0, 3, 1, 2)
    position[:, 1] = -position[:, 1] * aspect_ratio  # flip y axis

    if dim == 1:
        x, y = torch.split(position, 1, dim=1)
        position = x
    if dim == 3:

        x, y = torch.split(position, 1, dim=1)

        z = torch.ones_like(x) * torch.rand(1, device=device) * 2 - 1

        a = torch.randint(0, 3, (1,)).item()
        if a == 0:
            xyz = [x, y, z]
        elif a == 1:
            xyz = [z, x, y]
        else:
            xyz = [x, z, y]

        position = torch.cat(xyz, dim=1)

    position = position.expand(batch_size, dim, height, width)

    return position


def transform_coord(coord, t_coeff, dim):
    device = 'cuda' if coord.is_cuda else 'cpu'
    identity_matrix = torch.nn.init.eye_(torch.empty(dim, dim, device=device))

    bs, octaves, h, w, dim = coord.size()

    t_coeff = t_coeff.reshape(bs, octaves, dim, dim).unsqueeze(2).unsqueeze(2)

    t_coeff = t_coeff.expand(bs, octaves, h, w, dim, dim)
    t_coeff = t_coeff.reshape(bs * octaves, h, w, dim, dim)

    transform_matrix = identity_matrix.expand(bs * octaves, dim, dim)
    transform_matrix = transform_matrix.unsqueeze(1).unsqueeze(1)
    transform_matrix = transform_matrix.expand(bs * octaves, h, w, dim, dim)

    transform_matrix = transform_matrix + t_coeff
    transform_matrix = transform_matrix.reshape(h * w * bs * octaves, dim, dim)

    coord = coord.reshape(h * w * bs * octaves, dim, 1)
    coord_transformed = torch.bmm(transform_matrix, coord).squeeze(2)
    coord_transformed = coord_transformed.reshape(bs, octaves, h, w, dim)

    return coord_transformed