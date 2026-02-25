import torch
from hopfield_lib.hopfield_network_dream import HopfieldNetworkDream
from hopfield_lib.utils import bruitage_pixels_fixe, show_results

image_size = 16
patterns = torch.randint(0, 2, (250, image_size, image_size)).float() * 2 - 1

net = HopfieldNetworkDream(image_size=image_size, epsilon=0.005, order='async', max_iter=30)

net.train_Hebb(patterns)
net.dreaming_projector_matrix(nb_iter=10000)
noisy = bruitage_pixels_fixe(patterns[:2].clone(), 0)
show_results(noisy, net, targets=patterns[:2])