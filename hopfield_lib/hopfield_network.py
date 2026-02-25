import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from hopfield_lib.utils import distance_hamming
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class HopfieldNetwork:
    """Hopfield Network class."""

    def __init__(self, image_size, activation_function=torch.sign, order='sync', max_iter=300):
        """Initialize the object."""
        self.image_size = image_size
        self.nb_neurones = image_size * image_size
        self.w_matrix = torch.zeros(self.nb_neurones, self.nb_neurones)
        if activation_function == torch.sign:
            self.activation_function = lambda x: torch.where(x >= 0, 1, -1)
        else:
            self.activation_function = activation_function
        self.order = order
        self.max_iter = max_iter
        self.train_data = None
        self.trained = False

    def train_Hebb(self, X, null_diagonal=True):
        """Run train  Hebb."""
        self.train_data = X
        training_data = torch.reshape(X, [-1, self.nb_neurones]).float().to(DEVICE)
        self.w_matrix = 1 / self.nb_neurones * (training_data.T @ training_data)
        if null_diagonal:
            self.w_matrix.fill_diagonal_(0)
        self.trained = True

    def reset(self):
        """Run reset."""
        self.w_matrix = torch.zeros(self.nb_neurones, self.nb_neurones)
        self.trained = False

    def get_w_matrix(self):
        """Run get w matrix."""
        return self.w_matrix.to('cpu')

    def get_training_data(self):
        """Run get training data."""
        if self.trained:
            return self.train_data.to(DEVICE)
        else:
            return ValueError("Réseau non entrainer impossible de récupérer les données d'entrainement")

    def save(self, path_matrix='./w_matix.pt', path_training_data='./training_data.pt'):
        """Run save."""
        if self.trained:
            torch.save(self.train_data, path_training_data)
        torch.save(self.w_matrix, path_matrix)

    def load(self, trained=False, path_matrix='./w_matix.pt', path_training_data='./training_data.pt'):
        """Run load."""
        if trained:
            self.trained = trained
            self.train_data = torch.load(path_training_data).to(DEVICE)
        self.w_matrix = torch.load(path_matrix).to(DEVICE)

    def _calculate_energy(self, img, w_matrix):
        """Run calculate energy."""
        if img.device != self.w_matrix.device:
            img = img.reshape(-1, self.nb_neurones).float().to(DEVICE)
        if img.shape[0] != self.nb_neurones:
            img = img.T
        energies = -0.5 * torch.sum(img * (w_matrix @ img), dim=0)
        return energies

    def calculate_energy(self, img):
        """Run calculate energy."""
        return self._calculate_energy(img, self.w_matrix)

    def resize_entry(self, entry):
        """Run resize entry."""
        shape = entry.shape
        if len(shape) < 3:
            entry = entry.unsqueeze(0)
            shape = entry.shape
        img = entry.reshape(-1, self.nb_neurones).float().to(DEVICE)
        img = img.transpose(0, -1)
        return (img, shape)

    def _apply(self, entry, w_matrix, equal=True):
        """Run apply."""
        img, shape = self.resize_entry(entry)
        for iteration in range(self.max_iter):
            image_next = img.clone()
            if self.order == 'async':
                indices = torch.randperm(self.nb_neurones, device=DEVICE)
                for i in range(self.nb_neurones):
                    idx = indices[i]
                    w_row = w_matrix[idx]
                    h_i = w_row @ image_next
                    new_val = self.activation_function(h_i)
                    image_next[idx] = new_val
            else:
                image_next = self.activation_function(w_matrix @ img).float()
            if equal and image_next.equal(img):
                break
            img = image_next
        energy = self._calculate_energy(img, w_matrix)
        img = img.transpose(0, -1)
        return (img.reshape(-1, shape[-2], shape[-1]), energy)

    def apply(self, entry):
        """Run apply."""
        return self._apply(entry, self.w_matrix)

    def apply_with_trajectory(self, entry):
        """Run apply with trajectory."""
        img, shape = self.resize_entry(entry)
        trajectory = [img.clone()]
        energies = self.calculate_energy(img).unsqueeze(1)
        cycle = False
        i = 0
        while i < self.max_iter:
            image_next = img.clone()
            if self.order == 'async':
                for idx in torch.randperm(self.nb_neurones):
                    h_i = self.w_matrix[idx] @ image_next
                    image_next[idx] = self.activation_function(h_i)
                    trajectory.append(image_next.clone())
                    energies = torch.cat([energies, self.calculate_energy(image_next).unsqueeze(1)], dim=1)
            else:
                image_next = self.activation_function(self.w_matrix @ img).float()
                trajectory.append(image_next.clone())
                energies = torch.cat([energies, self.calculate_energy(image_next).unsqueeze(1)], dim=1)
            if image_next.equal(img):
                break
            if i > 1:
                traj_stack = torch.stack(trajectory, dim=0)
                comparison = traj_stack == image_next
                identical_images = comparison.all(dim=1)
                has_cycled_mask = identical_images.any(dim=0)
                if has_cycled_mask.any():
                    cycle = True
                    trajectory = trajectory[:-1]
                    energies = energies[:, :-1]
                    break
            img = image_next
            i = i + 1
        trajectory = torch.stack(trajectory, dim=0).permute(2, 0, 1)
        img = img.transpose(0, -1)
        return (img.reshape(-1, shape[-2], shape[-1]), trajectory.reshape(-1, trajectory.shape[1], shape[1], shape[2]), energies, cycle)

    def _test_number_pattern_unstable(self, dataset, w_matrix):
        """Run test number pattern unstable."""
        result, _ = self._apply(dataset, w_matrix)
        result = result.reshape((-1, self.image_size, self.image_size))
        dataset = dataset.reshape((-1, self.image_size, self.image_size))
        hamming_result = distance_hamming(result, dataset)
        hamming_result = torch.where(hamming_result != 0, torch.ones_like(hamming_result), hamming_result)
        return hamming_result

    def test_number_pattern_unstable(self, dataset):
        """Run test number pattern unstable."""
        return self._test_number_pattern_unstable(dataset, self.w_matrix)

    def benchmark(self, dataset, train=True, path=None, k_step=1):
        """Run benchmark."""
        percentages = []
        if self.trained and train:
            print('Les poids du réseau vont être modifiés')
        for k in tqdm(range(1, len(dataset), k_step), desc='Calcul de la stabilité sur le data set'):
            if train:
                self.train_Hebb(dataset[:k])
            if k == 1:
                percentages.append(self.test_number_pattern_unstable(dataset[:k]).sum().item() / k)
            else:
                percentages.append(self.test_number_pattern_unstable(dataset[:k]).diag().sum().item() / k)
        self.visualize_benchmark(percentages, path)

    def visualize_benchmark(self, percentages, path=None):
        """Run visualize benchmark."""
        plt.plot(torch.arange(1, len(percentages) + 1), percentages)
        plt.xlabel('Nombre de patterns (k)')
        plt.ylabel('Fraction de patterns instables (%)')
        if max(percentages) > 0:
            plt.ylim(0, 1.05)
        plt.grid(True)
        if path is not None:
            plt.savefig(path)
        plt.show()
