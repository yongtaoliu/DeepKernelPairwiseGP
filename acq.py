"""
Acquisition functions for Deep Kernel Pairwise GP.
"""
import random
import torch
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption


def acq_ucb(dkl_model, X_pool, previous_comparisons=None, 
                            top_k=100, beta=2.0, strategy='max_ucb'):
    """
    Get next comparison pair using UCB (Upper Confidence Bound) acquisition.
    
    UCB balances exploitation (high predicted utility) and exploration (high uncertainty)
    by computing: UCB(x) = mean(x) + beta * std(x)
    
    Parameters
    ----------
    dkl_model : DeepKernelPairwiseGP
        Deep kernel model with feature extractor
    X_pool : torch.Tensor or np.ndarray
        Candidate pool of shape (n_candidates, input_dim)
    previous_comparisons : set of tuples, optional
        Set of (idx1, idx2) tuples representing already-asked comparisons
    top_k : int, optional (default=100)
        Number of candidates to consider (for efficiency)
    beta : float, optional (default=2.0)
        Exploration parameter. Higher beta = more exploration
        - beta=0: Pure exploitation (highest predicted utility)
        - beta→∞: Pure exploration (highest uncertainty)
        - beta=2: Balanced (recommended)
    strategy : str, optional (default='max_ucb')
        Strategy for selecting pairs:
        - 'max_ucb': Compare top UCB vs second top UCB
        - 'max_vs_uncertain': Compare top UCB vs most uncertain
        - 'top_ucb_diverse': Compare top UCB with diverse high-UCB points
    
    Returns
    -------
    best_pair : tuple of (int, int)
        Indices of the two candidates to compare next
    """
    if not isinstance(X_pool, torch.Tensor):
        X_pool = torch.tensor(X_pool, dtype=torch.float64, 
                             device=next(dkl_model.parameters()).device)
    
    n = len(X_pool)
    
    print(f"\nUCB Acquisition (beta={beta}, strategy='{strategy}')")
    
    # Get predictions for all candidates
    with torch.no_grad():
        X_features = dkl_model.feature_extractor(X_pool)
        posterior = dkl_model.gp_model.posterior(X_features)
        means = posterior.mean.squeeze(-1)
        stds = torch.sqrt(posterior.variance.squeeze(-1))
    
    # Compute UCB scores
    ucb_scores = means + beta * stds
    
    # Statistics
    print(f"  Mean utility: {means.mean().item():.4f} ± {means.std().item():.4f}")
    print(f"  Mean uncertainty: {stds.mean().item():.4f} ± {stds.std().item():.4f}")
    print(f"  UCB range: [{ucb_scores.min().item():.4f}, {ucb_scores.max().item():.4f}]")
    
    # Pre-filter to top-k by UCB for efficiency
    top_ucb_idx = torch.argsort(ucb_scores, descending=True)[:min(top_k, n)]
    
    selected_points = None
    best_value = float('-inf')
    evaluated_pairs = 0
    skipped_duplicates = 0
    skipped_previous = 0
    
    if strategy == 'max_ucb':
        # Strategy 1: Compare top UCB vs second top UCB
        # This focuses on finding the best among top candidates
        for i_idx, i in enumerate(top_ucb_idx):
            for j in top_ucb_idx[i_idx + 1:]:
                i_val, j_val = i.item(), j.item()
                
                # Skip identical points
                if torch.allclose(X_pool[i_val], X_pool[j_val], atol=1e-6):
                    skipped_duplicates += 1
                    continue
                
                # Skip previously compared pairs
                if previous_comparisons is not None:
                    if (i_val, j_val) in previous_comparisons or (j_val, i_val) in previous_comparisons:
                        skipped_previous += 1
                        continue
                
                # Score this pair: prefer pairs with high combined UCB
                pair_score = ucb_scores[i_val] + ucb_scores[j_val]
                
                evaluated_pairs += 1
                
                if pair_score > best_value:
                    best_value = pair_score
                    selected_points = (i_val, j_val)
    
    elif strategy == 'max_vs_uncertain':
        # Strategy 2: Compare highest UCB vs most uncertain
        # This validates the top candidate against uncertain ones
        top_idx = top_ucb_idx[0].item()
        uncertain_idx = torch.argsort(stds, descending=True)[:top_k]
        
        for j in uncertain_idx:
            j_val = j.item()
            
            if j_val == top_idx:
                continue
            
            # Skip identical points
            if torch.allclose(X_pool[top_idx], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue
            
            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (top_idx, j_val) in previous_comparisons or (j_val, top_idx) in previous_comparisons:
                    skipped_previous += 1
                    continue
            
            # Score: prefer high UCB (exploitation) + high uncertainty (exploration)
            pair_score = ucb_scores[top_idx] + stds[j_val]
            
            evaluated_pairs += 1
            
            if pair_score > best_value:
                best_value = pair_score
                selected_points = (top_idx, j_val)
    
    elif strategy == 'top_ucb_diverse':
        # Strategy 3: Compare top UCB with diverse high-UCB points
        # This explores among promising candidates
        top_idx = top_ucb_idx[0].item()
        
        for j in top_ucb_idx[1:]:
            j_val = j.item()
            
            # Skip identical points
            if torch.allclose(X_pool[top_idx], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue
            
            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (top_idx, j_val) in previous_comparisons or (j_val, top_idx) in previous_comparisons:
                    skipped_previous += 1
                    continue
            
            # Score based on UCB difference (want meaningful comparisons)
            ucb_diff = abs(ucb_scores[top_idx] - ucb_scores[j_val])
            # Small difference = informative comparison
            pair_score = 1.0 / (ucb_diff + 0.1)  # Prefer close UCB values
            
            evaluated_pairs += 1
            
            if pair_score > best_value:
                best_value = pair_score
                selected_points = (top_idx, j_val)

    print(f"  Evaluated {evaluated_pairs} pairs")
    print(f"  Skipped {skipped_duplicates} duplicate points")
    print(f"  Skipped {skipped_previous} previously compared pairs")
    
    # Validation
    if selected_points is None:
        raise RuntimeError(
            f"Could not find valid comparison pair!\n"
            f"Evaluated: {evaluated_pairs}, "
            f"Skipped duplicates: {skipped_duplicates}, "
            f"Skipped previous: {skipped_previous}"
        )
    
    if selected_points[0] == selected_points[1]:
        print (f"Same index selected twice! best_pair={selected_points}")
        return [selected_points[0]]
    
    if torch.allclose(X_pool[selected_points[0]], X_pool[selected_points[1]], atol=1e-6):
        print (f"Identical points selected! Indices: {selected_points}")
        return [selected_points[0]]
    
    # Detailed diagnostics
    i, j = selected_points
    print(f"Selected pair: ({i}, {j})")
    
    return selected_points

def acq_eubo (dkl_model, X_pool, previous_comparisons=None, top_k=100):
    """
    Get next point(s) using BoTorch Expected Utility of Best Option acquisition with DKL.

    Parameters
    ----------
    dkl_model : DeepKernelPairwiseGP
        Deep kernel model with feature extractor
    X_pool : torch.Tensor or np.ndarray
        Candidate pool of shape (n_candidates, input_dim)
    previous_comparisons : set of tuples, optional
        Set of (idx1, idx2) tuples representing already-asked comparisons
    top_k : int, optional (default=100)
        Number of most uncertain candidates to consider

    Returns
    -------
    best_pair : tuple of (int, int)
        Indices of the two candidates to compare next
    """
    if not isinstance(X_pool, torch.Tensor):
        X_pool = torch.tensor(X_pool, dtype=torch.float64,  # Use double for consistency
                             device=next(dkl_model.parameters()).device)

    n = len(X_pool)

    # Pre-filter by uncertainty - work in FEATURE space
    with torch.no_grad():
        # Extract features for all candidates
        X_features = dkl_model.feature_extractor(X_pool)  # [n, feature_dim]

        # Get uncertainties in feature space
        posterior = dkl_model.gp_model.posterior(X_features)
        uncertainties = torch.sqrt(posterior.variance.squeeze(-1))

    top_uncertain_idx = torch.argsort(uncertainties, descending=True)[:min(top_k, n)]

    # Create acquisition function on the GP model (which operates in feature space)
    acq = AnalyticExpectedUtilityOfBestOption(pref_model=dkl_model.gp_model)

    best_value = float('-inf')
    selected_points = None
    evaluated_pairs = 0
    skipped_duplicates = 0
    skipped_previous = 0

    for i_idx, i in enumerate(top_uncertain_idx):
        for j in top_uncertain_idx[i_idx + 1:]:
            i_val, j_val = i.item(), j.item()

            # Skip identical points in ORIGINAL space
            if torch.allclose(X_pool[i_val], X_pool[j_val], atol=1e-6):
                skipped_duplicates += 1
                continue

            # Skip previously compared pairs
            if previous_comparisons is not None:
                if (i_val, j_val) in previous_comparisons or (j_val, i_val) in previous_comparisons:
                    skipped_previous += 1
                    continue

            # Extract features for this pair
            with torch.no_grad():
                features_i = X_features[i_val]  # Already computed above
                features_j = X_features[j_val]

            # Stack features for acquisition evaluation [1, 2, feature_dim]
            comparison_pair = torch.stack([features_i, features_j]).unsqueeze(0)

            with torch.no_grad():
                acq_value = acq(comparison_pair).item()

            evaluated_pairs += 1

            if acq_value > best_value:
                best_value = acq_value
                selected_points = (i_val, j_val)

    print(f"Evaluated {evaluated_pairs} pairs")
    print(f"Skipped {skipped_duplicates} duplicate points")
    print(f"Skipped {skipped_previous} previously compared pairs")

    # Validation
    if selected_points is None:
        raise RuntimeError(
            f"Could not find valid comparison pair!\n"
            f"Evaluated: {evaluated_pairs}, "
            f"Skipped duplicates: {skipped_duplicates}, "
            f"Skipped previous: {skipped_previous}"
        )

    if selected_points[0] == selected_points[1]:
        print(f"Same index selected twice: {selected_points}")
        return [selected_points[0]]

    if torch.allclose(X_pool[selected_points[0]], X_pool[selected_points[1]], atol=1e-6):
        print(f"Identical points selected: {selected_points}")
        return [selected_points[0]]

    print(f"Selected: indices {selected_points}")

    return selected_points

def get_user_preference(train_idx1, train_idx2, pair_num=1, total_pairs=1):
    """
    Get user preference between two options.

    Parameters
    ----------
    pool_idx1, pool_idx2 : int
        Pool indices of the two options
    train_idx1, train_idx2 : int
        Training indices
    pair_num : int
        Current pair number
    total_pairs : int
        Total number of pairs

    Returns
    -------
    preferred_train_idx, dispreferred_train_idx : int
        Training indices in order (winner, loser)
    """
    while True:
        choice = input(f"Pair {pair_num}/{total_pairs} → Which is better? Enter 0 or 1:\n").strip()
        if choice == "0":
            return train_idx1, train_idx2
        elif choice == "1":
            return train_idx2, train_idx1
        else:
            print("Invalid input. Please enter 0 or 1.")

def sample_comparison_pairs(train_indices, n_pairs_per_point=1, best_train_idx=None, seed=None):
    """
    Create comparison pairs for newly added points.

    Strategy:
    - If 2 new points added: compare them with each other + random old points
    - If 1 new point added with known best: compare new vs best + random old points

    Parameters
    ----------
    train_indices : np.ndarray
        All training indices (into full pool)
    n_pairs_per_point : int
        Number of random comparisons per new point
    best_train_idx : int, optional
        Training index of current best point (if known)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    pairs_log : list of tuples
        Pairs of training indices to compare
    """
    if seed is not None:
        random.seed(seed)

    n_train = len(train_indices)
    last_idx = n_train - 1  # Newest point
    old_indices = list(range(n_train - 2))  # All points except last 2

    pairs_log = []

    if best_train_idx is None:
        # Two new points added: compare them against each other
        second_last_idx = n_train - 2
        pairs_log.append((second_last_idx, last_idx))
        new_points = [second_last_idx, last_idx]
    else:
        # One new point added: compare against known best
        pairs_log.append((best_train_idx, last_idx))
        new_points = [last_idx]

    # Compare each new point against n_pairs random old points
    for new_idx in new_points:
        selected_old = random.sample(old_indices, min(n_pairs_per_point, len(old_indices)))
        for old_idx in selected_old:
            pairs_log.append((old_idx, new_idx))

    return pairs_log




