import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from hopfield_lib.hopfield_network import HopfieldNetwork, DEVICE
from hopfield_lib.utils import distance_hamming, bruitage_pixels_fixe

class HopfieldNetworkModern(HopfieldNetwork):
    """Hopfield Network Modern class."""

    def __init__(self, image_size, activation_function=torch.sign, f=lambda x: torch.pow(x, 2), max_iter=30, images=None):
        """Initialize the object."""
        super().__init__(image_size, activation_function, 'async', max_iter)
        if images is not None:
            self.set_training_data(images)
        else:
            self.train_data = None
            self.trained = False
        self.application_function_f = f

    def set_training_data(self, X):
        """Run set training data."""
        self.train_data = torch.reshape(X, [-1, self.nb_neurones]).float().to(DEVICE)
        self.trained = True

    def _calculate_energy(self, img):
        """Run calculate energy."""
        img, shape = self.resize_entry(img)
        datas = self.get_training_data()
        return -torch.sum(self.application_function_f(datas @ img), dim=0)

    def resize_entry(self, entry):
        """Run resize entry."""
        shape = entry.shape
        if len(shape) < 3 and shape[-1] != self.nb_neurones:
            entry = entry.unsqueeze(0)
            shape = entry.shape
        img = entry.reshape(-1, self.nb_neurones).float().to(DEVICE)
        img = img.transpose(0, -1)
        return (img, shape)

    def _apply(self, entry, equal=True):
        """Run apply."""
        img, shape = self.resize_entry(entry)
        training_datas = self.get_training_data()
        for iteration in range(self.max_iter):
            image_next = img.clone()
            indices = torch.randperm(self.nb_neurones, device=DEVICE)
            for i in range(self.nb_neurones):
                idx = indices[i]
                res_total = training_datas @ image_next
                res_index = training_datas[:, idx].view(-1, 1) @ image_next[idx, :].view(1, -1)
                result_function_application = self.application_function_f(training_datas[:, idx].view(-1, 1) + res_total - res_index)
                result_function_application_neg = self.application_function_f(-training_datas[:, idx].view(-1, 1) + res_total - res_index)
                result = self.activation_function(torch.sum(result_function_application - result_function_application_neg, dim=0))
                image_next[idx] = result
            if equal and image_next.equal(img):
                break
            img = image_next
        energy = self._calculate_energy(img)
        img = img.transpose(0, -1)
        return (img.reshape(-1, shape[-2], shape[-1]), energy)

    def apply(self, entry):
        """Run apply."""
        return self._apply(entry)

    def _calculate_energy(self, img, w_matrix=None):
        """Run calculate energy."""
        if img.device != self.train_data.device:
            img = img.to(self.train_data.device)
        if img.dim() == 1:
            img = img.unsqueeze(1)
        dot_product = torch.matmul(self.train_data, img)
        f_values = self.application_function_f(dot_product)
        return -torch.sum(f_values, dim=0)

    def calculate_energy(self, img):
        """Run calculate energy."""
        return self._calculate_energy(img, self.w_matrix)

    def apply_with_trajectory(self, entry):
        """Run apply with trajectory."""
        img, shape = self.resize_entry(entry)
        trajectory = [img.clone()]
        energies = self.calculate_energy(img).reshape(1, 1)
        cycle = False
        i = 0
        training_datas = self.get_training_data()
        while i < self.max_iter:
            image_next = img.clone()
            for idx in torch.randperm(self.nb_neurones):
                res_total = training_datas @ image_next
                res_index = training_datas[:, idx].view(-1, 1) @ image_next[idx, :].view(1, -1)
                result_function_application = self.application_function_f(training_datas[:, idx].view(-1, 1) + res_total - res_index)
                result_function_application_neg = self.application_function_f(-training_datas[:, idx].view(-1, 1) + res_total - res_index)
                result = self.activation_function(torch.sum(result_function_application - result_function_application_neg, dim=0))
                image_next[idx] = result
                trajectory.append(image_next.clone())
                energies = torch.cat([energies, self.calculate_energy(image_next).reshape(1, 1)], dim=1)
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

    def _test_number_pattern_unstable(self, test_dataset, train_dataset=None):
        """Run test number pattern unstable."""
        result, _ = self._apply(test_dataset)
        if train_dataset is None:
            dataset = self.get_training_data()
        else:
            dataset = train_dataset
        hamming_matrix = distance_hamming(result, dataset)
        n_inputs = hamming_matrix.shape[0]
        n_targets = hamming_matrix.shape[1]
        if n_inputs % n_targets != 0:
            raise ValueError(f"Erreur: Inputs ({n_inputs}) n'est pas un multiple de Targets ({n_targets})")
        n_levels = n_inputs // n_targets
        dists_3d = hamming_matrix.view(n_levels, n_targets, n_targets)
        relevant_distances = dists_3d.diagonal(dim1=1, dim2=2)
        failures = torch.where(relevant_distances != 0, torch.ones_like(relevant_distances), relevant_distances)
        return failures.flatten()

    def test_number_pattern_unstable(self, test_dataset, train_dataset=None):
        """Run test number pattern unstable."""
        return self._test_number_pattern_unstable(test_dataset, train_dataset)

    def benchmark(self, dataset, train=True, path=None, k_step=1):
        """Run benchmark."""
        percentages = []
        for k in tqdm(range(1, len(dataset), k_step), desc='Calcul de la stabilité sur le data set'):
            if train:
                self.set_training_data(dataset[:k])
            if k == 1:
                percentages.append(self.test_number_pattern_unstable(dataset[:k]).sum().item() / k)
            else:
                percentages.append(self.test_number_pattern_unstable(dataset[:k]).diag().sum().item() / k)
        self.visualize_benchmark(percentages, path)

    def benchmark_noise_robustness(self, dataset, max_size_batch=50000, noise_step=15, k_step=1, k_start=2, k_end=None, noise_start=0, noise_end=None, vectorized_mode=True):
        """Run benchmark noise robustness."""
        N = self.nb_neurones
        if noise_end is None:
            noise_end = N // 2
        noise_levels = np.linspace(noise_start, noise_end, noise_step, dtype=np.int32)
        num_levels = len(noise_levels)
        batch_per_level = int(max_size_batch // num_levels)
        if batch_per_level < 10:
            print(f'⚠️ Attention : Budget trop faible pour {num_levels} niveaux. Force à 10 img/lvl.')
            batch_per_level = 10
        if k_end is None:
            k_end = len(dataset)
        k_values = list(range(k_start, k_end, k_step))
        if len(dataset) not in k_values and len(dataset) <= k_end:
            k_values.append(len(dataset))
        global_matrix = []
        print(f'Taille globale du batch : {max_size_batch} ops')
        print(f'Résolution    : {num_levels} niveaux de bruit')
        dataset = dataset.float().to(DEVICE)
        torch.no_grad()
        for k in tqdm(k_values, desc='Progression (K)'):
            current_train_data = dataset[:k]
            self.set_training_data(current_train_data)
            indices = torch.randint(0, k, (batch_per_level,), device=DEVICE)
            targets_base = current_train_data[indices]
            error_rates_for_this_k = []
            if vectorized_mode:
                targets_full = targets_base.repeat(num_levels, 1, 1)
                noise_vals = torch.tensor(noise_levels, device=DEVICE).repeat_interleave(batch_per_level)
                noisy_full = bruitage_pixels_fixe(targets_full.clone(), noise_vals)
                reconstructed_full, _ = self._apply(noisy_full, equal=True)
                reconstructed_reshaped = reconstructed_full.reshape(num_levels, batch_per_level, -1)
                targets_reshaped = targets_full.reshape(num_levels, batch_per_level, -1)
                dot_product = torch.einsum('lbn,lbn->lb', reconstructed_reshaped, targets_reshaped)
                N = self.nb_neurones
                dist_h = (N - dot_product) / 2
                dist_h_inv = (N + dot_product) / 2
                min_dist_h = torch.minimum(dist_h, dist_h_inv)
                seuil_erreur = 0
                success_mask = min_dist_h <= seuil_erreur
                accuracies = success_mask.float().mean(dim=1)
                error_rates_for_this_k = ((1.0 - accuracies) * 100).cpu().tolist()
                mean_hamming_dist = min_dist_h.mean(dim=1).cpu().tolist()
            else:
                for nb_noise in noise_levels:
                    noise_tensor = torch.full((batch_per_level,), nb_noise, device=DEVICE)
                    noisy_batch = bruitage_pixels_fixe(targets_base.clone(), noise_tensor)
                    reconstructed, _ = self._apply(noisy_batch, equal=True)
                    reconstructed = reconstructed.reshape(targets_base.shape)
                    equality_mask = torch.all(torch.eq(torch.sign(reconstructed), torch.sign(targets_base)), dim=1)
                    acc = equality_mask.sum().item() / batch_per_level
                    error_rates_for_this_k.append((1.0 - acc) * 100)
            global_matrix.append(error_rates_for_this_k)
        self.visualize_noise_robustness(np.array(global_matrix), k_values, noise_levels)
        return (np.array(global_matrix), k_values, noise_levels)

    def visualize_noise_robustness(self, matrix, k_values, noise_levels, path=None):
        """Run visualize noise robustness."""
        noise_min, noise_max = (min(noise_levels), max(noise_levels))
        y_limit = self.nb_neurones // 2
        k_min, k_max = (min(k_values), max(k_values))
        full_y_range = np.arange(0, y_limit + 1)
        full_matrix = np.zeros((len(full_y_range), len(k_values)))
        for i, y in enumerate(full_y_range):
            if y < noise_min:
                full_matrix[i, :] = matrix[:, 0]
            elif y > noise_max:
                full_matrix[i, :] = matrix[:, -1]
            else:
                idx = np.abs(np.array(noise_levels) - y).argmin()
                full_matrix[i, :] = matrix[:, idx]
        plt.figure(figsize=(10, 6))
        cmap = 'RdYlGn_r'
        img = plt.imshow(full_matrix, cmap=cmap, vmin=0, vmax=100, aspect='auto', extent=[k_min, k_max, 0, y_limit], interpolation='nearest', origin='lower')
        plt.axhline(y=noise_min, color='black', linestyle='-', linewidth=1.5, alpha=0.8, label='Zone calculée')
        plt.axhline(y=noise_max, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
        plt.colorbar(img, label="Taux d'erreur (%)")
        plt.xlabel('Nombre de patterns stockés (k)')
        plt.ylabel('Niveau de bruit (pixels inversés)')
        plt.title(f"Analyse énergétique : Taille du bassins d'attraction")
        plt.legend(loc='upper right')
        if path is not None:
            plt.savefig(path)
        plt.show()
