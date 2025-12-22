"""
Utility functions for Deep Kernel Pairwise GP project.
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
from acq import get_user_preference

def get_grid_coords (img, step=1):
    """
    Generate coordinate grid for a single 2D image.
    
    Args:
        img: 2D numpy array
        step: distance between grid points
    
    Returns:
        N x 2 array of (y, x) coordinates
    """
    h, w = img.shape[:2]
    coords = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            coords.append([i, j])
    return np.array(coords)


def get_subimages(img, coordinates, window_size):
    """
    Extract subimages centered at given coordinates.
    
    Args:
        img: 2D or 3D numpy array (h, w) or (h, w, c)
        coordinates: N x 2 array of (y, x) coordinates
        window_size: size of square window to extract
    
    Returns:
        subimages: (N, window_size, window_size, channels) array
        valid_coords: coordinates where extraction succeeded
        valid_indices: indices of valid extractions
    """
    if img.ndim == 2:
        img = img[..., None]
    
    h, w, c = img.shape
    half_w = window_size // 2
    
    subimages = []
    valid_coords = []
    valid_indices = []
    
    for idx, (y, x) in enumerate(coordinates):
        # Check boundaries
        if (y - half_w >= 0 and y + half_w <= h and
            x - half_w >= 0 and x + half_w <= w):
            
            patch = img[y - half_w:y + half_w,
                       x - half_w:x + half_w, :]
            
            if patch.shape[0] == window_size and patch.shape[1] == window_size:
                subimages.append(patch)
                valid_coords.append([y, x])
                valid_indices.append(idx)
    
    return (np.array(subimages), 
            np.array(valid_coords), 
            np.array(valid_indices))

def plot_option(img_full, coords, spectra, v_step, pool_idx, train_idx=None):
    """
    Plot a single option for user comparison.

    Parameters
    ----------
    img_full : np.ndarray
        Full image
    coords : np.ndarray
        All coordinates
    spectra : np.ndarray
        All spectra
    v_step : np.ndarray
        Voltage steps
    pool_idx : tuple
        Indexes in the full pool
    train_idx : int, optional
        Index in training set (if applicable)
    option_label : str
        Label for the option
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), dpi=150)

    # Image with location markers for comparision
    ax1.imshow(img_full, origin='lower')
    ax1.scatter(coords[pool_idx[0], 1], coords[pool_idx[0], 0], marker='x', s=50, c='r')
    ax1.scatter(coords[pool_idx[1], 1], coords[pool_idx[1], 0], marker='x', s=50, c='k')
    ax1.axis("off")

    # Spectrum
    ax2.plot(v_step, spectra[coords[pool_idx[0], 0], coords[pool_idx[0], 1]], c = 'r', label = f"Opt 0; idx: pool-{pool_idx[0]}, train-{train_idx[0]}")
    ax2.plot(v_step, spectra[coords[pool_idx[1], 0], coords[pool_idx[1], 1]], c = 'k', label = f"Opt 1; idx: pool-{pool_idx[1]}, train-{train_idx[1]}")
    ax2.legend(loc = 1)
    plt.show()
    plt.pause(0.1)
    plt.close()


def plot_predictions(coords, y, coord_train, mean, var, step, total_steps):
    """
    Visualize ground truth, predicted mean, and predicted variance.

    Parameters
    ----------
    coords : np.ndarray
        All candidate coordinates, shape (n_candidates, 2)
    y : np.ndarray
        Ground truth values for all candidates
    coord_train : np.ndarray
        Training set coordinates
    mean : np.ndarray
        Predicted utility means
    var : np.ndarray
        Predicted utility variances
    step : int
        Current exploration step (1-indexed)
    total_steps : int
        Total number of exploration steps
    """
    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4), dpi=100)

    # Ground truth
    a = ax1.scatter(coords[:, 1], coords[:, 0], c=y, cmap="viridis")
    plt.colorbar(a, ax=ax1)
    ax1.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax1.set_title(f'Ground Truth (Step {step}/{total_steps})')
    ax1.set_aspect('equal')

    # Predicted utility mean
    b = ax2.scatter(coords[:, 1], coords[:, 0], c=mean, cmap="viridis")
    plt.colorbar(b, ax=ax2)
    ax2.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax2.set_title(f'Utility Mean (Step {step}/{total_steps})')
    ax2.set_aspect('equal')

    # Predicted variance
    c = ax3.scatter(coords[:, 1], coords[:, 0], c=var, cmap="viridis")
    plt.colorbar(c, ax=ax3)
    ax3.scatter(coord_train[:, 1], coord_train[:, 0], s=50, c='red', marker='x')
    ax3.set_title(f'Predicted Variance (Step {step}/{total_steps})')
    ax3.set_aspect('equal')

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    plt.close()

def acquire_preference(img_full, train_indices, comparison_pairs, coords, spectra, v_step, 
                       y_groundtruth=None, mode='human', confidence_factors=False):
    """
    Display all comparison pairs and acquire user preferences.

    Parameters
    ----------
    img_full : np.ndarray
        Full image
    train_indices : np.ndarray
        Training indices (into full pool)
    comparison_pairs : list of tuples
        Pairs of training indices to compare
    coords : np.ndarray
        All coordinates
    spectra : np.ndarray
        All targets
    v_step : np.ndarray
        Voltage steps
    y_groundtruth: np.ndarray
        Ground Truth for simulating experiment
    mode: str
        Human or simulated mode
    confidence_factors: list
        confidence factors.

    Returns
    -------
    new_comparisons : torch.Tensor
        Comparisons in format [winner_idx, loser_idx], shape (n_pairs, 2)
    """
    for pair_idx, (train_idx1, train_idx2) in enumerate(comparison_pairs):
        pool_idx1 = train_indices[train_idx1]
        pool_idx2 = train_indices[train_idx2]

        # Plot both options
        print(f"Comparison {pair_idx + 1}/{len(comparison_pairs)}")

        plot_option(img_full, coords, spectra, v_step,
                   pool_idx=[pool_idx1,pool_idx2], train_idx=[train_idx1,train_idx2])

        # Get preference
        if mode == 'simulated':
            if y_groundtruth is None:
                raise ValueError("Ground truth 'y' required for simulated mode")
            winner, loser, confidence = get_simulated_preference(
                train_idx1, train_idx2, train_indices, y_groundtruth,
                pair_num=pair_idx + 1,
                total_pairs=len(comparison_pairs),
                cconfidence_factors=confidence_factors)
        
        elif mode == 'human':
            winner, loser, confidence = get_user_preference(
                train_idx1, train_idx2,
                pair_num=pair_idx + 1,
                total_pairs=len(comparison_pairs),
                confidence_factors=confidence_factors)

        new_comparisons = [winner, loser]
        print(f"Recorded: train_idx {winner} > train_idx {loser}")

    return torch.tensor(new_comparisons, dtype=torch.long), torch.tensor(confidence, dtype=torch.float64)

def get_simulated_preference(train_idx1, train_idx2, train_indices, y_groundtruth, 
                            pair_num=1, total_pairs=1, confidence_factors=[0.5, 0.75, 1.0]):
    """
    Simulate user preference using ground truth.
    
    """
    pool_idx1 = train_indices[train_idx1]
    pool_idx2 = train_indices[train_idx2]
    
    y1 = y_groundtruth[pool_idx1]
    y2 = y_groundtruth[pool_idx2]
    
    # Determine true preference
    if y1 > y2:
        winner = train_idx1
        loser = train_idx2
        utility_diff = y1 - y2
    else:
        winner = train_idx2
        loser = train_idx1
        utility_diff = y2 - y1
    
    # Simulate confidence based on utility difference
    if confidence_factors is not None:
        # if utility_diff < 0.1:
        #     confidence = 0.5
        # elif utility_diff < 0.3:
        #     confidence = 0.75
        if y1 < 0.2 and y2 < 0.2:
            confidence = confidence_factors[0]
        elif utility_diff < 0.2:
            confidence = confidence_factors[1]
        else:
            confidence = confidence_factors[2]
    else:
        confidence = 1.0
    
    print(f"Pair {pair_num}/{total_pairs} simulated")
    print(f"train_idx {train_idx1}: y={y1:.4f}")
    print(f"train_idx {train_idx2}: y={y2:.4f}")
    print(f"{winner} > {loser} (confidence={confidence:.2f})")
    
    return winner, loser, confidence