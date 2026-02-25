"""
utils.py
--------
Fonctions utilitaires pour les réseaux de Hopfield :
  - Seed & reproductibilité
  - Transformation des données (ToSignedTensor)
  - Bruit & images aléatoires
  - Distance de Hamming
  - Visualisation (show_results, make_gif, paysage énergétique, plot_generic)
  - Import & traitement des datasets (ImageNet, corrélation)
  - Simulations comparatives
"""
import pickle
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import torch
from tqdm import tqdm
from torchvision import transforms
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed=42):
    """Run seed everything."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ToSignedTensor:
    """To Signed Tensor class."""

    def __call__(self, x):
        """Special method __call__."""
        x = transforms.ToTensor()(x)
        return torch.where(x == 0, -torch.ones_like(x), torch.ones_like(x))

def create_random_image(images=None, number=None, size=None, typed=torch.int8):
    """Run create random image."""
    if images is None and size is None:
        raise ValueError("Spécifier au moins 'images' ou 'size'")
    if images is not None:
        size = images.shape[-1]
    if number is not None and images is not None:
        images = images.repeat(number, 1, 1)
        number = images.shape[0]
    else:
        number = 1
    random_image = torch.randint(0, 2, [number, size, size], dtype=typed) * 2 - 1
    if images is not None:
        return torch.where(images == 1, images, random_image)
    return random_image

def bruitage_pixels_fixe(batch_images, nombre_pixels):
    """Run bruitage pixels fixe."""
    if batch_images.dim() == 4:
        B, C, H, W = batch_images.shape
        total_pixels = C * H * W
        batch_flat = batch_images.view(B, -1)
    else:
        B, H, W = batch_images.shape
        total_pixels = H * W
        batch_flat = batch_images.flatten(1)
    device = batch_images.device
    if isinstance(nombre_pixels, int):
        if nombre_pixels > total_pixels:
            raise ValueError(f'Erreur: {nombre_pixels} > taille image ({total_pixels})')
        prob_map = torch.rand(B, total_pixels, device=device)
        _, indices = torch.topk(prob_map, k=nombre_pixels, dim=1)
        mask_flat = torch.zeros(B, total_pixels, device=device)
        mask_flat.scatter_(1, indices, 1)
    else:
        if not torch.is_tensor(nombre_pixels):
            nombre_pixels = torch.tensor(nombre_pixels, device=device)
        if nombre_pixels.dim() == 0:
            nombre_pixels = nombre_pixels.view(1).expand(B)
        if nombre_pixels.shape[0] != B:
            raise ValueError(f'Le tenseur nombre_pixels doit avoir la taille du batch ({B}), reçu {nombre_pixels.shape}')
        prob_map = torch.rand(B, total_pixels, device=device)
        pixel_ranks = prob_map.argsort(dim=1, descending=True).argsort(dim=1)
        mask_flat = (pixel_ranks < nombre_pixels.view(-1, 1)).float()
    mask = mask_flat.view_as(batch_images)
    images_bruites = torch.where(mask == 1, -batch_images, batch_images)
    return images_bruites

def distance_hamming(inputs, references):
    """Run distance hamming."""
    inputs = inputs.to(DEVICE)
    references = references.to(DEVICE)
    if references.dim() == 2 and references.shape[0] == references.shape[1] and (references.shape[-1] == inputs.shape[-1]):
        references = references.flatten()
    if inputs.dim() == 2 and inputs.shape[0] == inputs.shape[1]:
        inputs = inputs.flatten()
    if inputs.dim() > 2:
        inputs = inputs.flatten(start_dim=1)
    if references.dim() > 2:
        references = references.flatten(start_dim=1)
    if inputs.dim() == 1:
        inputs = inputs.unsqueeze(0)
    if references.dim() == 1:
        references = references.unsqueeze(0)
    assert inputs.shape[1] == references.shape[1], f'Erreur dimension neurones: {inputs.shape[1]} vs {references.shape[1]}'
    N_neurons = inputs.shape[1]
    dot_product = inputs @ references.T
    dist_normal = (N_neurons - dot_product) / 2
    dist_inverted = (N_neurons + dot_product) / 2
    distances = torch.minimum(dist_normal, dist_inverted)
    return distances

def show_results(datas, network, targets=None, use_trajectory=False):
    """Run show results."""
    n = len(datas)
    n_cols = 3 if targets is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(2 * n_cols, 2 * n))
    if n == 1:
        axes = [axes]
    if use_trajectory:
        results, trajectories, energies, cycle = network.apply_with_trajectory(datas)
        energies = energies[:, -1]
    else:
        results, energies = network.apply(datas)
    results = results.to('cpu')
    energies = energies.cpu()
    for i, image in enumerate(datas):
        if targets is not None:
            reference_image = targets[i]
            col_idx_output = 1
        else:
            reference_image = image
            col_idx_output = 0
        E = energies[i].item()
        D = distance_hamming(results[i], reference_image)
        print(f'Image {i} → Énergie : {E:.4f} | Hamming = {D.item()}')
        axes[i][col_idx_output].imshow(image, cmap='binary')
        axes[i][col_idx_output].axis('off')
        axes[i][col_idx_output].set_title(f'Entrée {i}')
        if targets is not None:
            axes[i][0].imshow(reference_image, cmap='binary')
            axes[i][0].axis('off')
            axes[i][0].set_title(f'Cible {i}')
        ax_out = axes[i][-1]
        ax_out.imshow(results[i], cmap='binary')
        ax_out.axis('off')
        ax_out.set_title(f'Sortie {i}\nÉnergie={E:.3f}\nHamming={D.item()}')
    plt.tight_layout()
    plt.show()

def make_gif_from_trajectory(trajectory, energies, input_image=None, filename='hopfield.gif', interval=200):
    """Run make gif from trajectory."""
    n_cols = 3 if input_image is not None else 2
    fig, ax = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    idx_offset = 0
    if input_image is not None:
        ax[0].imshow(input_image, cmap='binary', vmin=-1, vmax=1)
        ax[0].set_title('Entrée bruitée')
        ax[0].axis('off')
        idx_offset = 1
    ax_energy = ax[idx_offset]
    ax_energy.plot(np.arange(len(energies)), energies, color='tab:blue')
    point, = ax_energy.plot([0], [energies[0]], 'ro', label=f'E = {energies[0]:.2f}')
    ax_energy.set_xlabel('Itération')
    ax_energy.set_ylabel('Énergie')
    ax_energy.legend(loc='upper right')
    ax_recon = ax[idx_offset + 1]
    img_display = ax_recon.imshow(trajectory[0], cmap='binary', vmin=-1, vmax=1)
    ax_recon.set_title('Reconstruction')
    ax_recon.axis('off')

    def update(frame):
        """Run update."""
        point.set_data([frame], [energies[frame]])
        point.set_label(f'E = {energies[frame]:.2f}')
        ax_energy.legend(loc='upper right')
        img_display.set_data(trajectory[frame])
        fig.suptitle(f'Évolution temporelle - Étape {frame + 1}')
        return [point, img_display]
    plt.tight_layout()
    ani = animation.FuncAnimation(fig, update, frames=len(trajectory), interval=interval, blit=False, repeat=False)
    ani.save(filename, writer='pillow')
    plt.close(fig)
    print(f'GIF sauvegardé sous {filename}')

def projection_2D(img_tensor, refs=None):
    """Run projection 2 D."""
    flat = ((img_tensor + 1) / 2).flatten()
    weights = torch.linspace(0, 1, steps=flat.numel())
    proj_normal = torch.dot(flat, weights).item()
    proj_inverse = torch.dot(1 - flat, weights).item()
    return min(proj_normal, proj_inverse)

def projection_hamming(img_tensor, refs, echelle=1.0):
    """Run projection hamming."""
    d_vers_0 = distance_hamming(img_tensor, refs[0])
    d_vers_500 = distance_hamming(img_tensor, refs[1])
    total_dist_relative = d_vers_0 + d_vers_500
    if total_dist_relative == 0:
        return 0.0
    position_normalisee = d_vers_0 / total_dist_relative
    return (position_normalisee * echelle).item()

def plot_energetic_landscape(images, hopfield_network, number_noisy=10, projection_function=projection_2D):
    """Run plot energetic landscape."""
    training_data_set = hopfield_network.get_training_data()
    train_image_proj = [projection_function(image, training_data_set) for image in training_data_set]
    pattern_colors = ['#1f77b4', '#d62728']
    parasite_color = '#7f7f7f'
    labels_seen = set()
    all_energies = []
    all_trajectories_projected = []
    for i, image in tqdm(enumerate(images), desc='Calcul des trajectoires'):
        random_data = create_random_image(image, number_noisy)
        result, trajectories, energies_batch, cycle = hopfield_network.apply_with_trajectory(random_data)
        for k in range(number_noisy):
            trajectory_k = trajectories[k].cpu()
            energies_k = energies_batch[k].cpu()
            all_energies.append(energies_k.tolist())
            projected_path = [projection_function(img, training_data_set) for img in trajectory_k]
            all_trajectories_projected.append(projected_path)
    plt.figure(figsize=(12, 7))
    for i in range(len(all_trajectories_projected)):
        path = all_trajectories_projected[i]
        energies = all_energies[i]
        final_proj = path[-1]
        assigned_color = parasite_color
        current_label = 'État parasite'
        for j in range(len(train_image_proj)):
            if abs(final_proj - train_image_proj[j]) < 1e-05:
                assigned_color = pattern_colors[j] if j < len(pattern_colors) else plt.cm.tab10(j)
                current_label = f'Pattern {j + 1}'
                break
        if current_label not in labels_seen:
            plt.plot(path, energies, color=assigned_color, alpha=0.6, label=current_label, linewidth=2, zorder=2 if 'Pattern' in current_label else 1)
            labels_seen.add(current_label)
        else:
            plt.plot(path, energies, color=assigned_color, alpha=0.3, linewidth=1, zorder=1)
    plt.title('Paysage énergétique : Convergence vers les attracteurs', fontsize=14)
    plt.xlabel("Position sur l'axe de projection", fontsize=12)
    plt.ylabel('Énergie du réseau', fontsize=12)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title='Attracteurs finaux', frameon=True, shadow=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def normalize_energies(energies):
    """Run normalize energies."""
    E_min = np.min(energies)
    E_max = np.max(energies)
    if E_max - E_min == 0:
        return np.zeros_like(energies)
    return (energies - E_min) / (E_max - E_min)

def get_label_and_style(key):
    """Run get label and style."""
    cmap = cm.get_cmap('tab10')
    if key == 'exp':
        return ('Exp', cmap(10))
    try:
        k = int(key.split('_')[1])
        label = f'Polynôme $x^{{{k}}}$'
        color = cmap(k - 2)
        return (label, color)
    except:
        return (key, 'black')

def run_simulation(update_func, image_start, ref_images, max_iter=10):
    """Run run simulation."""
    network = HopfieldNetworkModern(16, f=update_func, max_iter=max_iter, images=ref_images)
    result, trajectory, energies, cycle = network.apply_with_trajectory(image_start)
    dists = distance_hamming(trajectory.reshape(-1, 16, 16), ref_images)[:, 0]
    energies_val = energies[0].cpu().numpy()
    return (energies_val, dists)

def plot_generic(data_dict, x_type, y_type, title, ylabel, invert_x=False):
    """Run plot generic."""
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel, fontsize=12)
    xlabel_map = {'iter': 'Itérations', 'dist': 'Distance de Hamming', 'energy': 'Énergie Normalisée'}
    plt.xlabel(xlabel_map.get(x_type, x_type), fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    if invert_x:
        plt.gca().invert_xaxis()
    for key, (energies, dists) in data_dict.items():
        norm_energies = normalize_energies(energies)
        y_data = dists if y_type == 'dist' else norm_energies
        if x_type == 'iter':
            x_data = range(len(y_data))
        elif x_type == 'dist':
            x_data = dists
        else:
            x_data = norm_energies
        label, color = get_label_and_style(key)
        marker = 'o' if x_type != 'iter' else None
        markersize = 4 if x_type != 'iter' else None
        plt.plot(x_data, y_data, label=label, color=color, linewidth=2, marker=marker, markersize=markersize)
    plt.legend(loc='upper right' if not invert_x else 'best', fontsize=15)
    plt.tight_layout()
    plt.show()

def run_hopfield_simulation(steps, update_function, base_image, ref_images):
    """Run run hopfield simulation."""
    results = []
    target_images = ref_images
    for k in tqdm(steps, desc='Simulation'):
        random_image = bruitage_pixels_fixe(base_image, int(k))[0]
        network = HopfieldNetworkModern(16, f=update_function, max_iter=10, images=ref_images)
        result, trajectory, energies, cycle = network.apply_with_trajectory(random_image)
        dists = distance_hamming(trajectory.reshape(-1, 16, 16), target_images)[:, 0]
        results.append((dists, energies[0].cpu().numpy(), k, result.detach().cpu()))
    return results

def importImageNet(path, image_size):
    """Run import Image Net."""
    with open(path, 'rb') as fo:
        ImageNet = pickle.load(fo)
    labels = np.array(ImageNet['labels'])
    unique_labels, indices = np.unique(labels, return_index=True)
    indices = np.array(indices)
    datas = torch.from_numpy(np.array(ImageNet['data'], dtype=np.float32)[indices])
    datas = datas.reshape(len(indices), 3, image_size, image_size)
    grayscale_datas = datas.mean(axis=1)
    threshold = 255 / 2
    return torch.where(grayscale_datas > threshold, torch.ones_like(grayscale_datas), -torch.ones_like(grayscale_datas))

def extractIndependant(dataset):
    """Run extract Independant."""
    dataset_flat = dataset.reshape([-1, 16 * 16])
    matrice_calcul = dataset_flat.T.cpu().numpy()
    Q, R, P = scipy.linalg.qr(matrice_calcul, pivoting=True)
    tol = 1e-05
    rang = np.sum(np.abs(np.diag(R)) > tol)
    indices_independants = P[:rang]
    return dataset[np.sort(indices_independants)]

def compute_pattern_correlation(patterns):
    """Run compute pattern correlation."""
    P, N = patterns.shape
    patterns_centered = patterns - patterns.mean(dim=1, keepdim=True)
    std = patterns_centered.std(dim=1, keepdim=True)
    patterns_normalized = patterns_centered / (std + 1e-08)
    corr_matrix = patterns_normalized @ patterns_normalized.T / N
    mask = 1 - torch.eye(P, device=patterns.device)
    off_diag_corr = corr_matrix * mask
    correlations = torch.abs(off_diag_corr[mask.bool()])
    stats = {'max': correlations.max().item(), 'abs_mean': correlations.abs().mean().item(), 'corr_matrix': off_diag_corr.cpu().numpy()}
    return stats

def analyze_dataset_correlation(patterns, plot=True):
    """Run analyze dataset correlation."""
    stats = compute_pattern_correlation(patterns)
    print(f'\n--- Statistiques de corrélation ---')
    print(f"Max:               {stats['max']:7.4f}")
    print(f"Moyenne |corr|:    {stats['abs_mean']:7.4f}")
    print(f'\n--- Interprétation ---')
    if stats['abs_mean'] < 0.1:
        print('Patterns TRÈS PEU CORRÉLÉS (quasi-aléatoires)')
    elif stats['abs_mean'] < 0.3:
        print('Patterns MODÉRÉMENT CORRÉLÉS')
    else:
        print('Patterns FORTEMENT CORRÉLÉS')
    if plot:
        fig, axes = plt.subplots(1, 1, figsize=(16, 4))
        im = axes.imshow(stats['corr_matrix'], cmap='Purples', vmin=0, vmax=1, aspect='auto')
        axes.set_xlabel('Pattern index')
        axes.set_ylabel('Pattern index')
        axes.set_title('Matrice de corrélation')
        plt.colorbar(im, ax=axes)
        plt.tight_layout()
        plt.show()
