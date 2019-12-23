import numpy as np
import torch

R = 4
ITER_NUM = 300

class FractalGen():
    def __init__(self):
        pass

    def gen_fractal(self, c, Z):
        c = torch.tensor(c, dtype=torch.complex32)
        zs = torch.tensor(Z)
        xs = torch.full(size=zs.size(), fill_value=c)
        ns = torch.zeros_like(zs)

        for i in range(0, ITER_NUM):
            zs_next = torch.where(torch.abs(zs) < R, zs**2 + xs, zs)
            not_diverged = torch.abs(zs_next) < R
            ns_next = ns + torch.tensor(not_diverged, torch.float32)
            zs = zs_next
            ns = ns_next

        final_z = zs
        final_step = ns

        def get_rgb(bg_ratio, ratio, final_z, final_step):
            if torch.abs(final_z) < R:
                return torch.tensor([0, 0, 0])

            v = torch.log2(final_step + R - torch.log2(torch.log2(torch.abs(final_z)))) / 5.

            if v < 1.0:
                return torch.tensor([v**bg_ratio[0], v**bg_ratio[1], v**bg_ratio[2]])
            else:
                v = max(0, 2.-v)
                return torch.tensor(v**ratio[0], v**ratio[1], v**ratio[2])

        channels = get_rgb((4, 2.5, 1), (0.9, 0.9, 0.9), final_z, final_step)

        return channels.permute((1, 2, 0))

if __name__ == '__main__':
    g = FractalGen()
    Z = np.mgrid[-1:1:2/1920, -1:1:2/1080]
    g.gen_fractal(1 + 1j, Z)