import torch
from torch import nn

class Model():
    def __init__(self, device=None):
        self.device = device
        self._build_model()

        learn_rate=0.0001

        def get_params(module_list):
            for module in module_list:
                for param in module.parameters():
                    yield param

        self._train_frame_optim = torch.optim.Adam(get_params(self._frame_layers), lr=learn_rate)
        self._train_loss_optim = torch.optim.Adam(get_params(self._loss_layers), lr=learn_rate)

    def get_frame(self, x):
        """ Pass in numpy array with shape [batches, 3, 1080, 1920] for `x`
            Returns numpy array with shape [batches, 3, 1080, 1920]
        """
        x = torch.tensor(x, device=self.device, requires_grad=False)

        for l in self._frame_layers:
            l.to(self.device)
            l.requires_grad = False
            l.zero_grad()
            l.eval()
        for l in self._frame_layers:
            x = l(x)

        torch.cuda.synchronize(device=self.device)

        return torch.Tensor.cpu(x).detach().numpy()

    def train_frame(self, x, y):
        """ Pass in numpy array with shape [batches, 3, 1080, 1920] for `x` and `y`
            Returns loss value
        """

        x = torch.tensor(x, device=self.device, requires_grad=True)
        y = torch.tensor(y, device=self.device, requires_grad=False)

        # forward pass
        for l in self._frame_layers:
            l.to(self.device)
            l.requires_grad = True
            l.zero_grad()
            l.train()
        for l in self._frame_layers:
            x = l(x)

        # backward pass
        y_pred = x
        y_target =  y

        loss_stack = nn.functional.mse_loss(y_pred, y_target).reshape([1])

        loss_weights = [x ** 3 for x in [1.3, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65]]

        for l in self._loss_layers:
            l.to(self.device)
            l.requires_grad = False
            l.zero_grad()
            l.eval()
        for li in range(0, len(self._loss_layers)):
            l = self._loss_layers[li]
            y_pred = l(y_pred)
            y_target = l(y_target)
            if li in [2, 5, 8, 11, 14, 17, 20, 23]:
                loss_weight = loss_weights[int((li-1)/3)]
                loss = nn.functional.mse_loss(y_pred, y_target) * loss_weight
                loss_stack = torch.cat([loss_stack, loss.reshape([1])], dim=0)

        final_loss = loss_stack.sum()
        final_loss.backward()

        self._train_frame_optim.step()

        torch.cuda.synchronize(device=self.device)

        return final_loss

    def train_loss(self, x, y, update_grads=False):
        """ `x` is a numpy array of shape [batches, 3, 1080, 1920], representing a 3-color channel image
            `y` is a numpy array of shape [batches], with a 0 or 1 for all elements indicating if `x` is the category of image we are learning
            Returns loss value
        """
        x = torch.tensor(x, device=self.device, requires_grad=True)

        # forward pass
        for l in self._loss_layers:
            l.to(self.device)
            l.requires_grad = True
            l.zero_grad()
            l.train()
        for l in self._loss_layers:
            x = l(x)

        # backward pass
        y_pred = x
        y_target = torch.tensor(y, device=self.device, requires_grad=False, dtype=torch.int64)
        loss = nn.functional.cross_entropy(y_pred, y_target)
        loss.backward()

        self._train_loss_optim.step()

        torch.cuda.synchronize(device=self.device)

        return loss

    def save(self, path):
        torch.save({
            'frame_params': [l.state_dict() for l in self._frame_layers],
            'loss_params': [l.state_dict() for l in self._loss_layers]
        }, path)

    def load(self, path):
        params_dict = torch.load(path)
        
        for i in range(0, len(params_dict['frame_params'])):
            sd = params_dict['frame_params'][i]
            self._frame_layers[i].load_state_dict(sd)

        for i in range(0, len(params_dict['loss_params'])):
            sd = params_dict['loss_params'][i]
            self._loss_layers[i].load_state_dict(sd)

    def _build_model(self):
        def conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=(1, 1), padding=1):
            return [
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            ]

        def conv2d_down(in_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=1):
            return [
                nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            ]

        def deconv2d_up(in_dim, out_dim):
            return [
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            ]

        # define our frame layers
        self._frame_layers = [
            # in [batch, 3, 1080, 1920]
            *conv2d(3, 8),
            *conv2d_down(8, 16),
            *conv2d_down(16, 32),
            ResNet(32),
            ResNet(32),
            ResNet(32),
            ResNet(32),
            ResNet(32),
            *deconv2d_up(32, 16),
            *deconv2d_up(16, 8),
            nn.Conv2d(8, 3, kernel_size=3, stride=1, padding=1)
            # out [batch, 3, 1080, 1920]
        ]

        # define our loss layers
        self._loss_layers = [
            # in [batch, 3, 1080, 1920] - Layer Index 0
            *conv2d(3, 4),
            *conv2d_down(4, 4),

            # in [batch, 4, 540, 960] - Layer Index 6
            *conv2d(4, 8),
            *conv2d_down(8, 8), # layers[10] -> relu 2_2

            # in [batch, 8, 270, 480] - Layer Index 12
            *conv2d(8, 16),
            *conv2d_down(16, 16),

            # in [batch, 16, 135, 240] - Layer Index 18
            *conv2d(16, 32),
            *conv2d_down(32, 32), # layers[22] -> relu 4_2

            # in [batch, 32, 68, 120] - Layer Index 24
            *conv2d(32, 64),
            *conv2d_down(64, 64),

            # in [batch, 64, 34, 60] - Layer Index 30
            *conv2d(64, 128),
            *conv2d_down(128, 128), # layers[34] -> relu 6_2

            # in [batch, 128, 17, 30] - Layer Index 36
            *conv2d(128, 256),
            *conv2d_down(256, 256),

            # in [batch, 256, 9, 15] - Layer Index 42
            *conv2d(256, 256),
            *conv2d_down(256, 256), # layers[46] -> relu 8_2

            # in [batch, 256, 5, 8] - Layer Index 48
            *conv2d(256, 256),
            *conv2d_down(256, 256), # layers[52] -> relu 9_2

            # in [batch, 256, 3, 4] - Layer Index 54
            Flatten(),

            # in [batch, 3072]
            nn.Linear(3072, 512),

            # in [batch, 256] - Layer Index 56
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.30),

            # in [batch, 512] - Layer Index 59
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.30),

            # in [batch, 512] - Layer Index 62
            nn.Linear(512, 2),
            nn.Linear(2, 2),
            nn.Softmax(dim=1)
        ]

class ResNet(nn.Module):
    def __init__(self, filters):
        super(ResNet, self).__init__()

        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(filters)
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=(3, 3), stride=(1, 1), padding=1)

        self.add_module('relu', self.relu)
        self.add_module('batch_norm', self.batch_norm)
        self.add_module('conv1', self.conv1)
        self.add_module('conv2', self.conv2)

    def forward(self, x):
        original_x = x
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        x = original_x + x
        return self.relu(x)


class NestedConcat(nn.Module):
    def __init__(self, name, inner_layers):
        super(NestedConcat, self).__init__()

        self._inner_layers = inner_layers

        ind = 1
        for m in inner_layers:
            self.add_module(f'{name}-{ind}', m)
            ind += 1

    def forward(self, x):
        original_x = x
        for l in self._inner_layers:
            x = l(x)

        return torch.cat((x, original_x), dim=1)

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)
