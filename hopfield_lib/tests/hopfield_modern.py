import torch
from hopfield_lib.hopfield_network_modern import HopfieldNetworkModern
from hopfield_lib.utils import bruitage_pixels_fixe, show_results

image_size = 16
patterns = torch.randint(0, 2, (100, image_size, image_size)).float() * 2 - 1

net = HopfieldNetworkModern(f = lambda x: torch.pow(x, 10), image_size=image_size, max_iter=20, images=patterns)
noisy = bruitage_pixels_fixe(patterns[:3].clone(), 25)
show_results(noisy, net, targets=patterns[:3])