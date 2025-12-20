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


class DeepKernelPairwiseGP(nn.Module):
    """
    PairwiseGP with deep kernel learning for high-dimensional inputs.
    """
    def __init__(
        self,
        datapoints,
        comparisons,
        input_dim,
        feature_dim=2,
        hidden_dims=[256, 128, 64],
        jitter=1e-4
    ):
        super().__init__()

        self.feature_extractor = ImageFeatureExtractor(
            input_dim=input_dim,
            feature_dim=feature_dim,
            hidden_dims=hidden_dims
        )

        self.feature_extractor = self.feature_extractor.to(
            device=datapoints.device,
            dtype=datapoints.dtype)

        with torch.no_grad():
            train_features = self.feature_extractor(datapoints)

        covar_module = ScaleKernel(
            RBFKernel(ard_num_dims=feature_dim)
        )

        self.gp_model = PairwiseGP(
            datapoints=train_features,
            comparisons=comparisons,
            covar_module=covar_module,
            input_transform=Normalize(d=feature_dim),
            jitter=jitter
        )

        self.train_datapoints = datapoints
        self.feature_dim = feature_dim
        self.input_dim = input_dim

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
    feature_dim=2,
    hidden_dims=[256, 128, 64],
    num_epochs=200,
    lr_features=1e-4,
    lr_gp=1e-2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    verbose=True
):
    """Train PairwiseGP with deep kernel learning."""
    if not isinstance(datapoints, torch.Tensor):
        datapoints = torch.from_numpy(datapoints).double()
    else:
        datapoints = datapoints.double()

    if not isinstance(comparisons, torch.Tensor):
        comparisons = torch.from_numpy(comparisons).long()
    else:
        comparisons = comparisons.long()

    datapoints = datapoints.to(device)
    comparisons = comparisons.to(device)

    model = DeepKernelPairwiseGP(
        datapoints=datapoints,
        comparisons=comparisons,
        input_dim=input_dim,
        feature_dim=feature_dim,
        hidden_dims=hidden_dims
    ).to(device)

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': lr_features},
        {'params': model.gp_model.parameters(), 'lr': lr_gp}
    ])

    mll = PairwiseLaplaceMarginalLogLikelihood(
        model.gp_model.likelihood,
        model.gp_model
    )

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
        loss = -mll(output, model.gp_model.train_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if verbose and (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    model.eval()
    return model, losses


def fit_dkpg(X_train, train_comp, num_epochs=2000):
    """
    Fit Deep Kernel PairwiseGP model.

    Parameters
    ----------
    X_train : np.ndarray
      High-dimensional features (N, D) - flattened image patches
    train_comp : np.ndarray
      Pairwise comparisons (M, 2)
    num_epochs : int
        Number of training epochs

    Returns
    -------
    mll : PairwiseLaplaceMarginalLogLikelihood
        Marginal log likelihood
    dkl_model : DeepKernelPairwiseGP
        Trained deep kernel model
    """

    print("Training Deep Kernel PairwiseGP Model")

    input_dim = X_train.shape[-1]

    dkl_model, losses = train_dkpg(
        datapoints=X_train,
        comparisons=train_comp,
        input_dim=input_dim,
        feature_dim=2,
        hidden_dims=[256, 128, 64],
        num_epochs=num_epochs,
        lr_features=1e-4,
        lr_gp=1e-2,
        verbose=True)

    pref_model = dkl_model.gp_model
    mll = PairwiseLaplaceMarginalLogLikelihood(pref_model.likelihood, pref_model)

    # plt.figure(figsize=(4, 2))
    # plt.plot(losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Negative MLL')
    # plt.title('Training Loss')
    # plt.show()

    return mll, pref_model, dkl_model

def predict_utility(model, test_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict utility for test data."""
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
