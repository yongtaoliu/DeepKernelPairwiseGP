# Deep Kernel Pairwise GP for Active Learning

Deep Kernel Learning with Pairwise Gaussian Processes for active learning with preference feedback from experts.

- **Deep Kernel Learning (DKL)**: Neural network feature extractor + Gaussian Process
- **Pairwise Comparisons**: Learn from relative preferences ("A is better than B")
- **Active Learning**: Intelligently select which pairs to compare

The system learns to predict a scalar target value (e.g., preference score) from high-dimensional inputs (e.g., image patches) using only pairwise comparison feedback.



## Usage

### Basic Workflow
!pip install git+https://github.com/yongtaoliu/DeepKernelPairwiseGP.git

```python
from model import fit_dkpg, predict_utility
from acq import acq_eubo, get_user_preference, sample_comparison_pairs
from utils import get_grid_coords, get_subimages, plot_option, plot_predictions, acquire_preference

# 1. Prepare your data
X = ...  # High-dimensional features (n_samples, n_features)
train_indices = ...  # Initial training indices
train_comp = ...  # Initial comparisons (n_comparisons, 2)

# 2. Train the model
mll, pref_model, dkl_model = fit_dkpg(
    X_train=X[train_indices], 
    train_comp=train_comp,
    num_epochs=1000
)

# 3. Make predictions
mean, var = predict_utility(dkl_model, X)

# 4. Select next pair to compare (acquisition function)
selected_pair = acq_eubo(
    dkl_model=dkl_model,
    X_pool=X,
    previous_comparisons=set(),
    top_k=100
)

# 5. Get user preference and update training data
# ... then repeat from step 2
```

### Acquisition Functions

Two acquisition functions are available:

#### 1. EUBO (Expected Utility of Best Option)
```python
from acq import acq_eubo

selected_pair = acq_eubo(
    dkl_model=dkl_model,
    X_pool=X,
    previous_comparisons=previous_comparisons,
    top_k=100
)
```

- Uses BoTorch's `AnalyticExpectedUtilityOfBestOption`
- Maximizes expected information gain
- Good for finding the best option

#### 2. UCB (Upper Confidence Bound)
```python
from acq import acq_ucb

selected_pair = acq_ucb(
    dkl_model=dkl_model,
    X_pool=X,
    previous_comparisons=previous_comparisons,
    top_k=100,
    beta=2.0,
    strategy='max_ucb'
)
```

- Balances exploration and exploitation
- `beta` controls exploration (higher = more exploration)
- Three strategies available:
  - `'max_ucb'`: Compare top UCB points
  - `'max_vs_uncertain'`: Compare best vs uncertain
  - `'top_ucb_diverse'`: Compare among top UCB points


## Citation

If you use this code, please cite:
```bibtex
@software{DeepKernelPairwiseGP,
  title = {Deep Kernel Pairwise GP for Active Learning},
  author = {Yongtao Liu},
  email = {youngtaoliu@gmail.com},
  year = {2025},
  url = {https://github.com/yongtaoliu/DeepKernelPairwiseGP}
}
```

