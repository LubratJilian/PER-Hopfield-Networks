import torch
from hopfield_lib.hopfield_network import HopfieldNetwork
from hopfield_lib.utils import bruitage_pixels_fixe, make_gif_from_trajectory

image_size = 16
patterns = torch.randint(0, 2, (2, image_size, image_size)).float() * 2 - 1

net = HopfieldNetwork(image_size=image_size, order='async', max_iter=100)
net.train_Hebb(patterns)

noisy = bruitage_pixels_fixe(patterns[:1].clone(), 40)

result, trajectory, energies, cycle = net.apply_with_trajectory(noisy)

print("Cycle détecté :", cycle)
print("Énergie finale :", energies[0, -1].item())

make_gif_from_trajectory(
    trajectory=trajectory[0].cpu(),
    energies=energies[0].cpu().numpy(),
    input_image=noisy[0].cpu(),
    filename="convergence.gif"
)