import torch
from tqdm import tqdm
from hopfield_lib.hopfield_network import HopfieldNetwork, DEVICE
from hopfield_lib.utils import distance_hamming

class HopfieldNetworkDream(HopfieldNetwork):
    """Hopfield Network Dream class."""

    def __init__(self, image_size, epsilon=0.005, activation_function=torch.sign, order='sync', max_iter=30):
        """Initialize the object."""
        super().__init__(image_size, activation_function, order, max_iter)
        self.epsilon = epsilon

    def _dreaming_unlearning(self, nb_image_random, w_matrices, alpha, beta):
        """Run dreaming unlearning."""
        batch_size = w_matrices.shape[0]
        mask_diag = 1 - torch.eye(self.nb_neurones, device=w_matrices.device)
        w_matrices = w_matrices * mask_diag.unsqueeze(0)
        nb_iters = torch.tensor([max(1, int(self.nb_neurones / (self.epsilon * nb_image_random) * (alpha * (k / self.nb_neurones) - beta))) for k in range(1, batch_size + 1)], device=w_matrices.device)
        max_iter = nb_iters.max().item()
        for iteration in tqdm(range(max_iter)):
            random_image = torch.randint(0, 2, [1, nb_image_random, self.nb_neurones], device=w_matrices.device).float() * 2 - 1
            random_image = random_image.expand(batch_size, -1, -1)
            results = self._apply_async_batch_w(random_image, w_matrices)
            correlation_terms = torch.einsum('bij,bik->bjk', results, results)
            correlation_matrices = self.epsilon * correlation_terms / self.nb_neurones
            update_mask = (iteration < nb_iters).float().view(batch_size, 1, 1)
            w_matrices = w_matrices - correlation_matrices * update_mask
            w_matrices = w_matrices * mask_diag.unsqueeze(0)
        return w_matrices

    def _apply_async_batch_w(self, entries, w_matrices):
        """Run apply async batch w."""
        img = entries.clone()
        img_prev = torch.empty_like(img)
        for iteration in range(self.max_iter):
            img_prev.copy_(img)
            indices = torch.randperm(self.nb_neurones, device=w_matrices.device)
            for i in range(self.nb_neurones):
                idx = indices[i]
                w_rows = w_matrices[:, idx, :]
                h = torch.einsum('bin,bn->bi', img, w_rows)
                img[:, :, idx] = self.activation_function(h)
            if torch.equal(img, img_prev):
                break
        return img

    def dreaming_unlearning(self, nb_images, alpha=1.2, beta=0.05):
        """Run dreaming unlearning."""
        self.w_matrix = self.w_matrix.unsqueeze(0)
        self.w_matrix = self._dreaming_unlearning(nb_images, self.w_matrix, alpha, beta).squeeze(0)

    def benchmark_unlearning(self, W_global, nb_image_random, alpha=1.2, beta=0.05):
        """Run benchmark unlearning."""
        return self._dreaming_unlearning(nb_image_random, W_global, alpha, beta)

    def calculate_local_fields(self, image):
        """Run calculate local fields."""
        return self.w_matrix @ image.T

    def _dreaming_projector_matrix(self, nb_iter, W_global):
        """Run dreaming projector matrix."""
        batch_size = W_global.shape[0]
        eigenvalues, _ = torch.linalg.eigh(W_global)
        epsilon_optimal = 0.9 / eigenvalues[:, -1]
        epsilons = epsilon_optimal.view(batch_size, 1, 1)
        for _ in tqdm(range(nb_iter), desc='Optimisation des matrices de poids W'):
            random_image = torch.randint(0, 2, [1, self.nb_neurones], dtype=torch.float, device=DEVICE) * 2 - 1
            random_image_batch = random_image.expand(batch_size, self.nb_neurones)
            random_image_batch = random_image_batch.unsqueeze(2).to(W_global.device)
            local_fields_batch = torch.bmm(W_global, random_image_batch)
            local_fields = local_fields_batch.squeeze(2)
            correlation_matrix_batch = torch.einsum('bi,bj->bij', local_fields, local_fields)
            dream_term = epsilons * correlation_matrix_batch / self.nb_neurones
            W_global = W_global - dream_term
        return W_global

    def dreaming_projector_matrix(self, nb_iter):
        """Run dreaming projector matrix."""
        self.train_data = self.resize_entry(self.train_data)[0].T
        self.w_matrix = torch.einsum('xi,xj -> xij', self.train_data, self.train_data).to(DEVICE)
        self.w_matrix = torch.sum(self.w_matrix, dim=0) / self.nb_neurones
        self.w_matrix = self._dreaming_projector_matrix(nb_iter, self.w_matrix.unsqueeze(0)).squeeze(0)

    def benchmark_projector_matrix(self, W_global, nb_iter):
        """Run benchmark projector matrix."""
        return self._dreaming_projector_matrix(nb_iter, W_global)

    def benchmark_dream(self, dataset, path=None, method=None, **kwargs):
        """Run benchmark dream."""
        if method is None:
            method = self.benchmark_unlearning
        if method not in [self.benchmark_unlearning, self.benchmark_projector_matrix]:
            raise ValueError(f"La méthode fournie n'est pas valide.")
        dataset, _ = self.resize_entry(dataset)
        dataset = dataset.T
        self.train_data = dataset
        W_k = torch.einsum('xi,xj -> xij', dataset, dataset).to(DEVICE)
        W_global = torch.cumsum(W_k, dim=0) / self.nb_neurones
        with torch.no_grad():
            W_global = method(W_global, **kwargs)
            self.w_matrix = W_global[-1].to(DEVICE)
        percentages = []
        for k in tqdm(range(1, len(dataset) + 1), desc='Calcul de la stabilité sur le data set'):
            if k == 1:
                percentages.append(self._test_number_pattern_unstable(dataset[:k], W_global[k - 1]).sum().item() / k)
            else:
                percentages.append(self._test_number_pattern_unstable(dataset[:k], W_global[k - 1]).diag().sum().item() / k)
        self.visualize_benchmark(percentages, path)
