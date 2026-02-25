import torch
from hopfield_lib.hopfield_network import HopfieldNetwork
from hopfield_lib.utils import bruitage_pixels_fixe, show_results, seed_everything

seed_everything(42)

image_size = 16
patterns = torch.randint(0, 2, (3, image_size, image_size)).float() * 2 - 1

net = HopfieldNetwork(image_size=image_size, order='sync', max_iter=30)
net.train_Hebb(patterns)

noisy = bruitage_pixels_fixe(patterns[:1].clone(), nombre_pixels=30)

show_results(noisy, net, targets=patterns[:1])