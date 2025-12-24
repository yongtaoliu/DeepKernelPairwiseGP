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
                       y_groundtruth=None, mode='human', confidence_factors=[0.5, 0.75, 1.0],
                       allow_ties=True, tie_threshold=0.1):
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
    allow_ties : bool, optional (default=False)
        If True, allow tie/equal comparisons
    tie_threshold : float, optional (default=0.1)
        Threshold for simulation (only if allow_ties=True)

    Returns
    -------
    new_comparisons : torch.Tensor
        Comparisons in format [winner_idx, loser_idx], shape (n_pairs, 2)
    """
    new_comparisons = []
    confidence_weights = []

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
            
            idx1, idx2, comp_type, confidence = get_simulated_preference(
                train_idx1, train_idx2, train_indices, y_groundtruth,
                tie_threshold=tie_threshold,
                confidence_factors=confidence_factors,
                allow_ties=allow_ties
            )
        
        elif mode == 'human':
            idx1, idx2, comp_type, confidence = get_user_preference(
                train_idx1, train_idx2,
                pair_num=pair_idx + 1,
                total_pairs=len(comparison_pairs),
                confidence_factors=confidence_factors,
                allow_ties=allow_ties
            )
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'human' or 'simulated'.")
        
        new_comparisons.append([idx1, idx2, comp_type])
        confidence_weights.append(confidence)
        print(f"Recorded: train_idx {idx1} > train_idx {idx2}")

    return torch.tensor(new_comparisons, dtype=torch.long), torch.tensor(confidence_weights, dtype=torch.float64)

def get_simulated_preference(train_idx1, train_idx2, train_indices, y_groundtruth, tie_threshold=0.1, 
                            confidence_factors=[0.5, 0.75, 1.0], allow_ties=True):
    """
    Simulate user preference using ground truth.

    Parameters
    ----------
    train_idx1, train_idx2 : int
        Training indices to compare
    train_indices : np.ndarray
        Mapping from training indices to pool indices
    y_groundtruth : np.ndarray
        Ground truth utilities
    tie_threshold : float, optional (default=0.1)
        If |y1 - y2| < tie_threshold, consider it a tie
        Only used if allow_ties=True
    confidence_factors : list of float
        Confidence levels for weak/medium/strong preferences
    allow_ties : bool, optional (default=False)
        If True, can return tie comparisons

    Returns
    -------
    idx1, idx2 : int
        Training indices
    comp_type : int
        0, 1, or 2
    confidence : float
        Confidence weight
    
    """
    pool_idx1 = train_indices[train_idx1]
    pool_idx2 = train_indices[train_idx2]
    
    y1 = y_groundtruth[pool_idx1]
    y2 = y_groundtruth[pool_idx2]
    
    utility_diff = abs(y1 - y2)

    # Determine comparison type based on ground truth
    if y1<0.2 or y2<0.2:
        true_confidence = confidence_factors[0]
        if allow_ties and utility_diff < tie_threshold:
            # They're close enough to be equal
            true_comp_type = 2
        elif y1 > y2:
            true_comp_type = 0
        else:
            true_comp_type = 1
    else:
        true_confidence = confidence_factors[2]
        if allow_ties and utility_diff < tie_threshold:
            # They're close enough to be equal
            true_comp_type = 2
        elif y1 > y2:
            true_comp_type = 0
        else:
            true_comp_type = 1
    
    if true_comp_type == 0:
        print(f"{train_idx1} > {train_idx2} (confidence={true_confidence:.2f})")
    elif true_comp_type == 1:
        print(f"{train_idx2} > {train_idx1} (confidence={true_confidence:.2f})")
    elif true_comp_type == 2:
        print(f"{train_idx1} ≈ {train_idx2} (equal, confidence={true_confidence:.2f})")
    
    return train_idx1, train_idx2, true_comp_type, true_confidence