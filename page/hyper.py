# hyper.py
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, embedding_model, embedding_output_size, num_weights, num_biases):
        super().__init__()
        self.embedding_model = embedding_model
        self.embedding_model_params = [p for p in embedding_model.parameters() if p.requires_grad]
        self.num_weights = num_weights
        self.num_biases = num_biases

        self.weights_gen = nn.Linear(embedding_output_size, num_weights)
        self.bias_gen = nn.Linear(embedding_output_size, num_biases)
        self.parameters_generators_input_size = embedding_output_size

    def calc_variance4init(self, main_net_in_size, train_dataloader, hyper_input_type,
                           embd_vars=False, main_net_relu=True, main_net_biasses=True, var_hypernet_input=None):

        if var_hypernet_input is None:
            variances = []

            for batch in iter(train_dataloader):

                if hyper_input_type == "tabular":
                    values = batch[2]
                else:
                    values = batch[0]

                if embd_vars:
                    values = self.embedding_model(values)

                for v in values:
                    variances.append(np.array(v.view(-1).detach().cpu()).var())

            var_hypernet_input = np.mean(variances)
            if var_hypernet_input == 0:
                var_hypernet_input = 1

        dk = self.parameters_generators_input_size
        dj = main_net_in_size
        var_weights_generator = (2 ** main_net_relu) / ((2 ** main_net_biasses) * dj * dk * var_hypernet_input)
        var_biasses_generator = (2 ** main_net_relu) / (2 * dk * var_hypernet_input)
        return var_weights_generator, var_biasses_generator

    def variance_uniform_init(self, var_weights_generator, var_biasses_generator):

        ws_init = np.sqrt(3 * var_weights_generator)
        bs_init = np.sqrt(3 * var_biasses_generator)
        nn.init.uniform_(self.weights_gen.weight, -ws_init, ws_init)
        nn.init.uniform_(self.bias_gen.weight, -bs_init, bs_init)
        nn.init.constant_(self.weights_gen.bias, 0)
        nn.init.constant_(self.bias_gen.bias, 0)

    def initialize_parameters(self, weights_init_method, fan_in, hyper_input_type,
                              train_loader=None, var_hypernet_input=None):

        if weights_init_method == "input_variance":
            print("HyperNetwork: input_variance initialization")
            var_w, var_b = self.calc_variance4init(fan_in, train_loader, hyper_input_type, embd_vars=False,
                                                   var_hypernet_input=var_hypernet_input)
            self.variance_uniform_init(var_w, var_b)
        elif weights_init_method == "embedding_variance":
            print("HyperNetwork: embedding_variance initialization")
            var_w, var_b = self.calc_variance4init(fan_in, train_loader, hyper_input_type, embd_vars=True)
            self.variance_uniform_init(var_w, var_b)
        else:
            raise ValueError("HyperNetwork initialization type not implemented!")

    def freeze_embedding_model(self):
        for p in self.embedding_model_params:
            p.requires_grad = False

    def unfreeze_embedding_model(self):
        for p in self.embedding_model_params:
            p.requires_grad = True

    def forward(self, x, return_emb=True):
        emb_out = self.embedding_model(x)
        weights = self.weights_gen(emb_out)
        biases = self.bias_gen(emb_out)
        if return_emb:
            return weights, biases, emb_out
        return weights, biases


class HyperLinearLayer(nn.Module):

    def __init__(self, in_features, out_features, embedding_model, embedding_output_size,
                 weights_init_method=None, train_loader=None, hyper_input_type=None):
        super().__init__()
        num_weights = in_features * out_features
        num_biases = out_features
        self.hyper_net = HyperNetwork(embedding_model, embedding_output_size, num_weights, num_biases)
        self.num_out_features = out_features
        self.weights_shape = (out_features, in_features)

        if weights_init_method is not None:
            self.hyper_net.initialize_parameters(weights_init_method, in_features, hyper_input_type,
                                                 train_loader=train_loader)

    def forward(self, x):
        # x: tuple (data_tensor, embedding_features)
        x_input, features = x[0], x[1]
        # print("features shape:", features.shape)
        # print("features sample:", features[0])

        weights, biases, emb_out = self.hyper_net(features, return_emb=True)
        out = torch.zeros((x_input.shape[0], self.num_out_features), dtype=x_input.dtype, layout=x_input.layout,
                          device=x_input.device)
        for i, (w, b) in enumerate(zip(weights, biases)):
            w = w.reshape(self.weights_shape)
            out[i] = F.linear(x_input[i], w, b)
        return out, emb_out


class LinearLayer(nn.Module):
    def __init__(self, in_features, out_features, embedding_model=None, embedding_output_size=None,
                 weights_init_method=None, train_loader=None, hyper_input_type=None):
        super().__init__()
        self.hyper = embedding_model is not None
        if self.hyper:
            self.layer = HyperLinearLayer(
                in_features, out_features, embedding_model, embedding_output_size,
                weights_init_method=weights_init_method, train_loader=train_loader,
                hyper_input_type=hyper_input_type
            )
        else:
            self.layer = nn.Linear(in_features, out_features)

    def forward(self, x):
        if self.hyper:
            return self.layer(x)
        else:
            # x assumed to be tensor if not hyper
            return self.layer(x), None

import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, embedding_model, embedding_output_size,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None, var_hypernet_input=None,
                 stride=1, padding=1):
        super().__init__()
        num_weights = in_channels * out_channels * (kernel_size ** 2)
        num_biases = out_channels
        self.stride = stride
        self.padding = padding

        self.hyper_net = HyperNetwork(embedding_model, embedding_output_size, num_weights, num_biases)

        self.num_out_channels = out_channels
        self.weights_shape = (out_channels, in_channels, kernel_size, kernel_size)

        if weights_init_method is not None:
            fan_in = in_channels * (kernel_size ** 2)
            self.hyper_net.initialize_parameters(weights_init_method, fan_in, hyper_input_type,
                                                train_loader=train_loader,
                                                 var_hypernet_input=var_hypernet_input)

    def forward(self, x):
        x, features = x[0], x[1]

        weights, biases, emb_out = self.hyper_net(features, return_emb=True)

        # first sample forward to determine shape
        out0 = F.conv2d(input=x[0][None], weight=weights[0].reshape(self.weights_shape),
                        bias=biases[0], stride=self.stride, padding=self.padding)

        out = torch.zeros([x.shape[0]] + list(out0.shape[1:]), dtype=x.dtype, layout=x.layout, device=x.device)
        out[0] = out0
        if x.shape[0] > 1:
            for i, (w, b) in enumerate(zip(weights[1:], biases[1:])):
                w = w.reshape(self.weights_shape)
                out[i + 1] = F.conv2d(input=x[i][None], weight=w, bias=b, stride=self.stride,
                                      padding=self.padding)
        return out, emb_out

class Conv2DLayer(nn.Module):
    def __init__(self,  in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 embedding_model=None, embedding_output_size=None,
                 weights_init_method=None, train_loader=None, hyper_input_type=None, GPU=None,
                 var_hypernet_input=None):
        super().__init__()
        self.hyper = embedding_model is not None
        if self.hyper:
            self.layer = HyperConv2dLayer(
                in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                embedding_model=embedding_model, embedding_output_size=embedding_output_size,
                weights_init_method=weights_init_method, train_loader=train_loader,
                hyper_input_type=hyper_input_type, GPU=GPU, var_hypernet_input=var_hypernet_input
            )
        else:
            self.layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x_tuple):
        if self.hyper:
            return self.layer(x_tuple)  # always returns (tensor, emb_out)
        else:
            x, features = x_tuple
            out = self.layer(x)
            return out, None

class HyperPreactivResBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1,
                 hyper_embedding_models=(None, None, None), **hyper_kwargs):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=bn_momentum)
        self.conv1 = Conv2DLayer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                 embedding_model=hyper_embedding_models[0], **hyper_kwargs)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        self.conv2 = Conv2DLayer(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                                 embedding_model=hyper_embedding_models[1], **hyper_kwargs)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = Conv2DLayer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               embedding_model=hyper_embedding_models[2], **hyper_kwargs)
            self.downsample_bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum)
        else:
            self.downsample_conv = None
            self.downsample_bn = None

    def forward(self, x_tuple):
        x, features = x_tuple

        if self.downsample_conv is not None:
            identity, _ = self.downsample_conv((x, features))
            identity = self.downsample_bn(identity)
        else:
            identity = x

        out, emb_out = self.conv1((x, features))
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out, _ = self.conv2((out, features))
        out += identity

        return out, emb_out


class HyperPreactivResBlock2D_TTT(HyperPreactivResBlock2D):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        embedding_model = hyper_kwargs.pop("embedding_model")
        hyper_embedding_models = [embedding_model, embedding_model, embedding_model]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)

class HyperPreactivResBlock2D_TTF(HyperPreactivResBlock2D):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        embedding_model = hyper_kwargs.pop("embedding_model")
        hyper_embedding_models = [embedding_model, embedding_model, None]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)
class HyperPreactivResBlock2D_TTT(HyperPreactivResBlock2D):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        embedding_model = hyper_kwargs.pop("embedding_model")
        hyper_embedding_models = [embedding_model, embedding_model, embedding_model]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)
class HyperPreactivResBlock2DFFT(HyperPreactivResBlock2D):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, **hyper_kwargs):
        embedding_model = hyper_kwargs.pop("embedding_model")
        hyper_embedding_models = [None, None, embedding_model]
        super().__init__(in_channels, out_channels, bn_momentum=bn_momentum, dropout=dropout, stride=stride,
                         hyper_embedding_models=hyper_embedding_models, **hyper_kwargs)