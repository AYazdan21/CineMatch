# CineMatch 🎬

A sophisticated movie recommendation system built using collaborative filtering and matrix completion techniques. CineMatch implements an EM-based matrix factorization algorithm with SoftImpute regularization to predict user ratings and generate personalized movie recommendations.

## Overview

CineMatch tackles the classic collaborative filtering problem: given a sparse user-item rating matrix, predict missing ratings and recommend movies users are likely to enjoy. The system uses advanced matrix completion techniques that go beyond simple baseline models to capture latent factors in user preferences and movie characteristics.

## Features

- **Baseline Model**: User and item bias estimation with ridge regularization
- **Matrix Factorization**: Truncated SVD using power iteration method
- **EM Algorithm**: Expectation-Maximization for matrix completion with SoftImpute shrinkage
- **Hyperparameter Tuning**: Automatic selection of optimal rank (k) and regularization (λ)
- **Visualization**: Energy plots, factor interpretation, and PCA-based clustering
- **Performance Metrics**: RMSE tracking on both training reconstruction and test predictions

## Dataset

The project uses the MovieLens dataset with:
- **Training set**: 81,016 ratings from 943 users on 1,648 movies
- **Test set**: 8,976 ratings for evaluation
- **Rating scale**: 1.0 to 5.0 stars
- **Sparsity**: ~94% of the user-item matrix is unobserved

Each rating includes:
- `user_id`: Unique user identifier
- `item_id`: Unique movie identifier  
- `rating`: User's rating (1-5 stars)
- `title`: Movie title with release year

## Methodology

### 1. Baseline Model
Estimates user biases (`bu`) and item biases (`bi`) using alternating least squares with ridge regularization:
```
rating ≈ μ + bu[user] + bi[item]
```
where `μ` is the global mean rating.

### 2. Matrix Factorization
Decomposes the rating matrix into low-rank factors:
```
R ≈ U · Σ · V^T
```
using truncated SVD with power iteration for efficiency.

### 3. EM Matrix Completion
Iteratively refines the matrix factorization:
1. **E-step**: Impute missing values using current factorization
2. **M-step**: Compute SVD on the completed matrix
3. **Regularization**: Apply SoftImpute shrinkage to singular values
4. **Re-impose**: Restore observed ratings to prevent drift

The algorithm tracks both training reconstruction error and test RMSE to prevent overfitting.

### 4. Hyperparameter Optimization
- **Rank (k)**: Tests k=1 to k=20 to find optimal latent dimensionality
- **Regularization (λ)**: SoftImpute shrinkage parameter (default: 1.5)
- **Best performance**: Achieved at k=4 with Test RMSE ≈ 0.922

## Results

| Model | Test RMSE |
|-------|-----------|
| Baseline (biases only) | ~0.95 |
| **EM Matrix Completion (k=4, λ=1.5)** | **0.922** |
| Higher ranks (k>4) | 0.923-0.930 |

The optimal model uses only 4 latent factors, suggesting that user preferences can be effectively captured in a low-dimensional space (e.g., genre preferences, movie era, mainstream vs. niche).

## Project Structure

```
CineMatch/
├── final.ipynb          # Main notebook with complete pipeline
├── data.csv             # Training data (81K ratings)
├── test.csv             # Test data (9K ratings)
├── ranking_test.csv     # Additional ranking evaluation data
└── README.md            # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CineMatch.git
cd CineMatch

# Install dependencies
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Usage

1. **Open the notebook**:
```bash
jupyter notebook final.ipynb
```

2. **Run all cells** to execute the complete pipeline:
   - Data preprocessing and train/test split
   - Baseline model fitting
   - EM matrix completion with hyperparameter search
   - Visualization and interpretation

3. **Key parameters** (configurable in the notebook):
   - `DEFAULT_K = 4`: Number of latent factors
   - `DEFAULT_LAMBDA = 1.5`: SoftImpute shrinkage
   - `REG_BIAS = 10.0`: Ridge regularization for biases
   - `MAX_EM_ITER = 50`: Maximum EM iterations

## Algorithm Details

### Truncated SVD (Power Iteration)
Efficiently computes the top-k singular vectors without full SVD:
- Iteratively refines random initial vectors
- Converges to dominant singular values/vectors
- Much faster than full SVD for large sparse matrices

### SoftImpute Regularization
Applies soft-thresholding to singular values:
```
σ_shrunk = max(0, σ_raw - λ)
```
This prevents overfitting by shrinking small singular values toward zero.

### Convergence Criteria
The EM algorithm stops when:
- Maximum iterations reached (50)
- Test RMSE stops improving (early stopping)
- Reconstruction error stabilizes

## Visualization & Interpretation

The notebook includes:
- **RMSE vs. Rank**: Shows test error across different k values
- **Energy Plots**: Visualizes singular value spectrum and shrinkage
- **Factor Interpretation**: Analyzes what latent dimensions capture
- **PCA Clustering**: 2D/3D visualization of user and movie embeddings

## Performance Considerations

- **Memory**: Stores full U×M matrix (~943×1648 = 1.5M entries)
- **Computation**: Power iteration is O(k·n·iterations) per EM step
- **Scalability**: For larger datasets, consider stochastic methods or sparse representations

## Future Improvements

- [ ] Incorporate movie metadata (genres, directors, actors)
- [ ] Implement temporal dynamics (rating patterns over time)
- [ ] Add content-based filtering for cold-start users/items
- [ ] Deploy as REST API for real-time recommendations
- [ ] Experiment with neural collaborative filtering (NCF)
- [ ] Add implicit feedback signals (views, clicks)

## References

- Mazumder, R., Hastie, T., & Tibshirani, R. (2010). *Spectral Regularization Algorithms for Learning Large Incomplete Matrices*. JMLR.
- Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*. IEEE Computer.
- MovieLens Dataset: [GroupLens Research](https://grouplens.org/datasets/movielens/)

## Authors

Amirreza Yazdanpanah

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or feedback, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating collaborative filtering techniques. For production recommender systems, consider additional factors like scalability, real-time updates, and A/B testing infrastructure.
