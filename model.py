"""
Deep Kernel Pairwise GP model for preference learning.
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from gpytorch.kernels import RBFKernel, ScaleKernel

class ImageFeatureExtractor(nn.Module):
    """
    Feature extractor for image patches.
    """
    def __init__(self, input_dim, feature_dim=2, hidden_dims=[256, 128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, feature_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class ConfidenceWeightedMLL(nn.Module):
    """
    Marginal log likelihood with confidence weighting.
    """
    
    def __init__(self, likelihood, model, confidence_weights):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights
        if confidence_weights.sum() > 0:
            self.normalized_weights = confidence_weights / confidence_weights.sum() * len(confidence_weights)
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, target):
        """
        Compute weighted marginal log likelihood.
        """
        mean = output.mean
        comparisons = target
        
        total_weighted_ll = torch.tensor(0.0, dtype=torch.float64, device=mean.device)
        
        for i in range(len(comparisons)):
            winner_idx = comparisons[i, 0].long()
            loser_idx = comparisons[i, 1].long()
            
            # Utility difference
            mean_diff = mean[winner_idx] - mean[loser_idx]
            
            # Variance of difference
            var_winner = output.variance[winner_idx]
            var_loser = output.variance[loser_idx]
            var_diff = var_winner + var_loser
            
            # Log probability
            std_diff = torch.sqrt(var_diff + 1e-6)
            z_score = mean_diff / std_diff
            
            # Normal CDF
            normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
            log_prob = torch.log(normal_cdf + 1e-8)
            
            # Weight by confidence (all in float64)
            confidence = self.normalized_weights[i]
            weighted_log_prob = confidence * log_prob
            
            total_weighted_ll = total_weighted_ll + weighted_log_prob
        
        return total_weighted_ll

class ConfidenceWeightedMLLWithTies(nn.Module):
    """
    Marginal log likelihood with confidence weighting and tie support.
    """
    
    def __init__(self, likelihood, model, confidence_weights, tolerance=0.1):
        super().__init__()
        self.likelihood = likelihood
        self.model = model
        self.tolerance = tolerance
        
        if confidence_weights.dtype != torch.float64:
            confidence_weights = confidence_weights.double()
        self.confidence_weights = confidence_weights
        
        # Normalize weights
        if confidence_weights.sum() > 0:
            self.normalized_weights = confidence_weights / confidence_weights.sum() * len(confidence_weights)
        else:
            self.normalized_weights = confidence_weights
    
    def forward(self, output, comparisons):
        """
        Compute weighted marginal log likelihood with tie support.
        
        Parameters
        ----------
        output : gpytorch posterior
            GP posterior
        comparisons : torch.Tensor
            Comparisons with types, shape (n_comparisons, 3)
            [:, 0] = first point index
            [:, 1] = second point index
            [:, 2] = type (0=first>second, 1=second>first, 2=equal)
        """
        mean = output.mean
        variance = output.variance
        
        total_weighted_ll = torch.tensor(0.0, dtype=torch.float64, device=mean.device)
        
        for i in range(len(comparisons)):
            idx_a = comparisons[i, 0].long()
            idx_b = comparisons[i, 1].long()
            comp_type = comparisons[i, 2].long()
            
            # Utility difference
            mean_diff = mean[idx_a] - mean[idx_b]
            
            # Variance of difference
            var_diff = variance[idx_a] + variance[idx_b]
            std_diff = torch.sqrt(var_diff + 1e-6)
            
            if comp_type == 0:  # A > B
                # P(A > B) = Φ((mean_diff - tolerance) / std_diff)
                z_score = (mean_diff - self.tolerance) / std_diff
                normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                log_prob = torch.log(normal_cdf + 1e-8)
                
            elif comp_type == 1:  # B > A
                # P(B > A) = Φ((-mean_diff - tolerance) / std_diff)
                z_score = (-mean_diff - self.tolerance) / std_diff
                normal_cdf = 0.5 * (1 + torch.erf(z_score / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                log_prob = torch.log(normal_cdf + 1e-8)
                
            elif comp_type == 2:  # A ≈ B (equal/tie)
                # P(|diff| < tolerance) = Φ((tolerance - |mean_diff|) / std_diff) - Φ((-tolerance - |mean_diff|) / std_diff)
                abs_diff = torch.abs(mean_diff)
                z_upper = (self.tolerance - abs_diff) / std_diff
                z_lower = (-self.tolerance - abs_diff) / std_diff
                
                prob_upper = 0.5 * (1 + torch.erf(z_upper / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                prob_lower = 0.5 * (1 + torch.erf(z_lower / torch.sqrt(torch.tensor(2.0, dtype=torch.float64))))
                
                log_prob = torch.log(prob_upper - prob_lower + 1e-8)
            
            else:
                raise ValueError(f"Unknown comparison type: {comp_type}. Expected 0, 1, or 2.")
            
            # Weight by confidence
            confidence = self.normalized_weights[i]
            weighted_log_prob = confidence * log_prob
            total_weighted_ll = total_weighted_ll + weighted_log_prob
        
        return total_weighted_ll

class DeepKernelPairwiseGP(nn.Module):
    """
    PairwiseGP with deep kernel learning for high-dimensional inputs.
    """
    def __init__(
        self,
        datapoints,
        comparisons,
        input_dim,
        feature_dim=16,
        hidden_dims=[256, 128, 64],
        confidence_weights=None,
        jitter=1e-4
    ):
        super().__init__()

        self.feature_extractor = ImageFeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims)

        self.feature_extractor = self.feature_extractor.to(
            device=datapoints.device,
            dtype=datapoints.dtype)

        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)

        covar_module = ScaleKernel(RBFKernel(ard_num_dims=feature_dim))

        # PairwiseGP only accepts (n, 2) format: [idx1, idx2]
        # It doesn't understand comparison types (ties)
        
        if comparisons.shape[1] == 3:
            # We have comparison types (3 columns)
            # Extract only strict preferences (type 0 or 1)
            strict_mask = comparisons[:, 2] != 2  # Not ties
            strict_comparisons = comparisons[strict_mask, :2].clone()
            
            # Convert type 1 (second>first) to type 0 (first>second) by swapping
            type_1_mask = comparisons[strict_mask, 2] == 1
            strict_comparisons[type_1_mask] = strict_comparisons[type_1_mask].flip(1)
            
            # Store full comparisons for later use
            self.full_comparisons = comparisons
            self.has_ties = (comparisons[:, 2] == 2).any().item()
        else:
            # Standard 2-column format
            strict_comparisons = comparisons
            self.full_comparisons = comparisons
            self.has_ties = False

        self.gp_model = PairwiseGP(
            datapoints=train_features,
            comparisons=comparisons,
            covar_module=covar_module,
            input_transform=Normalize(d=feature_dim),
            jitter=jitter)

        self.train_datapoints = datapoints
        self.feature_dim = feature_dim
        self.input_dim = input_dim

        # Store confidence weights
        if confidence_weights is not None:
            if confidence_weights.dtype != torch.float64:
                confidence_weights = confidence_weights.double()
            self.confidence_weights = confidence_weights.to(datapoints.device)
        else:
            self.confidence_weights = torch.ones(
                len(comparisons), 
                dtype=torch.float64, 
                device=datapoints.device)
            
        self.register_buffer('_confidence_weights', self.confidence_weights)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.gp_model.posterior(features)

    def update_gp_data(self):
        features = self.feature_extractor(self.train_datapoints)
        self.gp_model.set_train_data(features, strict=False)


def train_dkpg(
    datapoints,
    comparisons,
    input_dim,
    feature_dim=16,
    hidden_dims=[256, 128, 64],
    confidence_weights=None,
    use_custom_mll=None,
    allow_ties=False,
    tolerance=0.1,
    num_epochs=1000,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
):
    """
    Train Deep Kernel PairwiseGP with flexible MLL selection.
    
    Parameters
    ----------
    datapoints : np.ndarray or torch.Tensor
        Training datapoints, shape (n, input_dim)
    comparisons : np.ndarray or torch.Tensor
        Pairwise comparisons, shape (m, 2)
    input_dim : int
        Input dimensionality
    feature_dim : int
        Learned feature dimensionality
    hidden_dims : list of int
        Hidden layer dimensions
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for comparisons, shape (m,)
        If provided, forces use of custom MLL
    use_custom_mll : bool, optional
        If True, use ConfidenceWeightedMLL (with weights=1.0 if not provided)
        If False, use standard PairwiseLaplaceMarginalLogLikelihood
        If None (default), auto-select:
            - Use custom if confidence_weights provided
            - Use standard otherwise
    allow_ties : bool
        If True, expect comparisons to have 3 columns with type information
        If False, expect standard 2-column format
    tolerance : float
        Tolerance for equal comparisons (only used if allow_ties=True)
    num_epochs : int
        Number of training epochs
    lr_features : float
        Learning rate for feature extractor
    lr_gp : float
        Learning rate for GP parameters
    device : str
        Device to use
    verbose : bool
        Print training progress
    
    Returns
    -------
    model : DeepKernelPairwiseGP
        Trained model
    losses : list
        Training losses
    """
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()

    if not isinstance(comparisons, torch.Tensor):
        comparisons = torch.from_numpy(comparisons).long()
    else:
        comparisons = comparisons.long()

    # Handle comparison format
    if allow_ties:
        if comparisons.shape[1] == 2:
            # Add type column (all type 0)
            types = torch.zeros(len(comparisons), 1, dtype=torch.long, device=comparisons.device)
            comparisons = torch.cat([comparisons, types], dim=1)
        
        # Count comparison types
        n_strict = ((comparisons[:, 2] == 0) | (comparisons[:, 2] == 1)).sum().item()
        n_ties = (comparisons[:, 2] == 2).sum().item()
    else:
        if comparisons.shape[1] == 3:
            comparisons = comparisons[:, :2]
        n_strict = len(comparisons)
        n_ties = 0

    # Confidence weights
    if confidence_weights is not None:
        if not isinstance(confidence_weights, torch.Tensor):
            confidence_weights = torch.from_numpy(confidence_weights).double()  
        else:
            confidence_weights = confidence_weights.double()  
        confidence_weights = confidence_weights.to(device)
        
        if verbose:
            print(f"confidence min: {confidence_weights.min()}, "
                  f"confidence max: {confidence_weights.max()}, ")
    else:
        confidence_weights = torch.ones(
            len(comparisons), 
            dtype=torch.float64,  
            device=device)

    datapoints = datapoints.to(device)
    comparisons = comparisons.to(device)

    # ===== MLL SELECTION LOGIC =====
    if use_custom_mll is None:
        # Auto-select: use custom if confidence weights are not all 1.0
        has_varying_confidence = not torch.allclose(
            confidence_weights, 
            torch.ones_like(confidence_weights))
        
        has_ties = allow_ties and n_ties > 0
        use_custom_mll = has_varying_confidence
        
        if verbose:
            if has_ties:
                print("Auto-selected: ConfidenceWeightedMLLWithTies (tie support)")
            elif has_varying_confidence:
                print("Auto-selected: Custom ConfidenceWeightedMLL (varying confidence weights)")
            else:
                print("Auto-selected: Standard PairwiseLaplaceMarginalLogLikelihood (equal weights)")
    else:
        if verbose:
            if use_custom_mll:
                mll_name = "ConfidenceWeightedMLLWithTies" if (allow_ties and n_ties > 0) else "ConfidenceWeightedMLL"
                print(f"User-selected: {mll_name}")
            else:
                print("User-selected: Standard PairwiseLaplaceMarginalLogLikelihood")

    model = DeepKernelPairwiseGP(
        datapoints=datapoints,
        comparisons=comparisons,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims,
        confidence_weights=confidence_weights
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_features},
        {'params': model.gp_model.parameters(), 'lr': lr_gp}
    ])

    # ===== SELECT MLL =====
    if use_custom_mll:
        if allow_ties and n_ties > 0:
            # Use tie-aware MLL
            mll = ConfidenceWeightedMLLWithTies(
                model.gp_model.likelihood,
                model.gp_model,
                confidence_weights,
                tolerance=tolerance)
            mll_name = "ConfidenceWeightedMLLWithTies"
            train_with_full_comparisons = True
        else:
            # Standard confidence-weighted MLL
            # Only use weights for strict comparisons
            if allow_ties and comparisons.shape[1] == 3:
                strict_mask = comparisons[:, 2] != 2
                conf_weights_strict = confidence_weights[strict_mask]
            else:
                conf_weights_strict = confidence_weights
                
            mll = ConfidenceWeightedMLL(
                model.gp_model.likelihood,
                model.gp_model,
                conf_weights_strict
            )
            mll_name = "ConfidenceWeightedMLL"
            train_with_full_comparisons = False
    else:
        # Standard BoTorch MLL
        mll = PairwiseLaplaceMarginalLogLikelihood(
            model.gp_model.likelihood,
            model.gp_model)
        mll_name = "PairwiseLaplaceMarginalLogLikelihood"
        train_with_full_comparisons = False

    model.train()
    losses = []

    if verbose:
        print(f"Training on {device}")
        print(f"Input dim: {input_dim}, Feature dim: {feature_dim}")
        print(f"Comparisons: {len(comparisons)}")

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        model.update_gp_data()
        output = model.gp_model(*model.gp_model.train_inputs)
        # Compute loss
        if train_with_full_comparisons:
            # Pass full comparisons (including ties) to tie-aware MLL
            loss = -mll(output, comparisons)
        else:
            # Pass only strict comparisons to standard MLL
            loss = -mll(output, model.gp_model.train_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    return model, losses


def fit_dkpg(X_train, train_comp, confidence_weights=None, 
             use_custom_mll=None, allow_ties=False, tolerance=0.1, 
             feature_dim=16, num_epochs=1000, verbose=True):
    """
    Fit Deep Kernel PairwiseGP model with optional confidence weighting.

    Parameters
    ----------
    X_train : np.ndarray or torch.Tensor
        High-dimensional features (N, D) - flattened image patches
    train_comp : np.ndarray or torch.Tensor
        Pairwise comparisons (M, 2), each row is [winner_idx, loser_idx]
    confidence_weights : np.ndarray or torch.Tensor, optional
        Confidence weights for each comparison, shape (M,)
        Values should be in [0, 1] where 1.0 = fully confident
        If provided, uses ConfidenceWeightedMLL
    use_custom_mll : bool, optional
        Explicitly choose MLL type:
        - True: Use ConfidenceWeightedMLL (with weights=1.0 if not provided)
        - False: Use standard PairwiseLaplaceMarginalLogLikelihood
        - None (default): Auto-select based on confidence_weights
    allow_ties : bool
        If True, support tie/equal comparisons
    tolerance : float
        Tolerance for considering utilities equal (only if allow_ties=True)
    feature_dim : int
        Dimensionality of learned feature space
    num_epochs : int
        Number of training epochs
    verbose : bool
        Print training information

    Returns
    -------
    mll : MarginalLogLikelihood
        Marginal log likelihood object (for reference)
    pref_model : PairwiseGP
        The GP model (operates in feature space)
    dkl_model : DeepKernelPairwiseGP
        Complete deep kernel model with feature extractor
    """

    print("Training Deep Kernel PairwiseGP Model")

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkpg(
        datapoints=X_train,
        comparisons=train_comp,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=[256, 128, 64],
        confidence_weights=confidence_weights,
        use_custom_mll=use_custom_mll,
        allow_ties=allow_ties,
        tolerance=tolerance,
        num_epochs=num_epochs,
        lr_features=1e-4,
        lr_gp=1e-2,
        verbose=verbose)

    pref_model = dkl_model.gp_model

   # Create MLL for reference (matches what was used in training)
    if allow_ties:
        # Check if there are actual ties
        if not isinstance(train_comp, torch.Tensor):
            train_comp_tensor = torch.from_numpy(train_comp).long()
        else:
            train_comp_tensor = train_comp
        
        has_ties = train_comp_tensor.shape[1] == 3 and (train_comp_tensor[:, 2] == 2).any()
        
        if has_ties:
            conf_weights = dkl_model.confidence_weights
            mll = ConfidenceWeightedMLLWithTies(
                pref_model.likelihood,
                pref_model,
                conf_weights,
                tolerance=tolerance)
        elif confidence_weights is not None or use_custom_mll:
            conf_weights = dkl_model.confidence_weights
            mll = ConfidenceWeightedMLL(
                pref_model.likelihood,
                pref_model,
                conf_weights)
        else:
            mll = PairwiseLaplaceMarginalLogLikelihood(
                pref_model.likelihood,
                pref_model)
    elif confidence_weights is not None or use_custom_mll:
        # Use same confidence weights as training
        conf_weights = dkl_model.confidence_weights
        mll = ConfidenceWeightedMLL(
            pref_model.likelihood, 
            pref_model,
            conf_weights)
    else: # Standard MLL
        mll = PairwiseLaplaceMarginalLogLikelihood(
            pref_model.likelihood, 
            pref_model)
        
    if verbose:
        plt.figure(figsize=(4, 2))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative MLL')
        plt.title('Training Loss')
        plt.show()

    return mll, pref_model, dkl_model

def predict_utility(model, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict utility for test data.
    model : DeepKernelPairwiseGP
        Trained model
    test_data : np.ndarray or torch.Tensor
        Test features, shape (n_test, input_dim)
    device : str
        Device to use
    
    Returns
    -------
    mean : np.ndarray
        Predicted utility means, shape (n_test,)
    uncertainty : np.ndarray
        Predicted variance, shape (n_test,)
    """
    
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.from_numpy(test_data).double()
    else:
        test_data = test_data.double()

    test_data = test_data.to(device)
    model.eval()

    with torch.no_grad():
        posterior = model(test_data)
        mean = posterior.mean.cpu().numpy()
        variance = posterior.variance.cpu().numpy()

    return mean, variance
