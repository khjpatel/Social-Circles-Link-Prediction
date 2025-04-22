import numpy as np
import sys
import os

def read_iris_data(filename):
    """
    Read iris dataset from file with more robust file handling.
    """
    try:
        # Check if file exists
        if not os.path.exists(filename):
            print(f"Error: File {filename} not found.")
            sys.exit(1)
        
        # Read data with more flexible delimiter handling
        data = np.genfromtxt("Users/khushipatel/Desktop/iris.data", delimiter=',')
        
        # Ensure data is loaded correctly
        if data.size == 0:
            print("Error: No data could be read from the file.")
            sys.exit(1)
        
        features = data[:, :-1]  # First 4 columns are features
        labels = data[:, -1]     # Last column is label
        return features, labels
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def initialize_parameters(features, k):
    """
    Initialize model parameters using first k groups of points.
    """
    n = len(features)
    group_size = n // k
    
    means = []
    covs = []
    
    for i in range(k):
        start = i * group_size
        end = (i + 1) * group_size if i < k-1 else n
        
        cluster_points = features[start:end]
        means.append(np.mean(cluster_points, axis=0))
        covs.append(np.cov(cluster_points.T))
    
    # Equal probability for all clusters
    weights = np.ones(k) / k
    
    return np.array(means), np.array(covs), weights

def gaussian_pdf(x, mu, cov):
    """
    Multivariate Gaussian Probability Density Function.
    """
    d = len(mu)
    try:
        # Ensure numerical stability
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        
        # Prevent zero or negative determinant
        if det <= 0:
            det = 1e-10
        
        norm_const = 1.0 / ((2 * np.pi) ** (d/2) * (det ** 0.5))
        diff = x - mu
        exponent = -0.5 * np.dot(np.dot(diff, inv), diff)
        
        return norm_const * np.exp(exponent)
    except Exception:
        return 0

def em_clustering(features, k, max_iter=1000, convergence_threshold=1e-6):
    """
    Expectation-Maximization Clustering Algorithm
    """
    n, d = features.shape
    
    # Initialize parameters
    means, covs, weights = initialize_parameters(features, k)
    
    for iteration in range(max_iter):
        # Store old means for convergence check
        old_means = means.copy()
        
        # E-step: Compute responsibilities
        responsibilities = np.zeros((n, k))
        for i in range(k):
            for j in range(n):
                responsibilities[j, i] = weights[i] * gaussian_pdf(features[j], means[i], covs[i])
        
        # Normalize responsibilities
        responsibilities /= responsibilities.sum(axis=1)[:, np.newaxis]
        
        # M-step: Update parameters
        N = responsibilities.sum(axis=0)
        
        # Update means
        means = np.dot(responsibilities.T, features) / N[:, np.newaxis]
        
        # Update covariance matrices
        covs = np.zeros_like(covs)
        for i in range(k):
            diff = features - means[i]
            covs[i] = np.dot((responsibilities[:, i] * diff.T), diff) / N[i]
        
        # Update cluster weights
        weights = N / n
        
        # Convergence check
        mean_distance = np.sum(np.linalg.norm(means - old_means, axis=1))
        if mean_distance < convergence_threshold:
            break
    
    # Assign final cluster membership
    cluster_membership = np.argmax(responsibilities, axis=1)
    
    return means, covs, cluster_membership, iteration + 1

def compute_purity(cluster_membership, true_labels):
    """
    Compute clustering purity
    """
    unique_labels = np.unique(true_labels)
    purity = 0
    
    for cluster in np.unique(cluster_membership):
        cluster_points = true_labels[cluster_membership == cluster]
        max_count = max([np.sum(cluster_points == label) for label in unique_labels])
        purity += max_count
    
    return purity / len(true_labels)

def main():
    # Hardcoded for direct running in IDE
    filename = '/Users/khushipatel/Desktop/DM1/iris.txt'
    k = 3
    
    # Read data
    features, true_labels = read_iris_data(filename)
    
    # Perform EM clustering
    means, covs, cluster_membership, iterations = em_clustering(features, k)
    
    # Compute purity
    purity = compute_purity(cluster_membership, true_labels)
    
    # Sort means by their norm
    mean_norms = np.linalg.norm(means, axis=1)
    sorted_indices = np.argsort(mean_norms)
    
    means = means[sorted_indices]
    covs = covs[sorted_indices]
    
    # Reorder cluster membership based on sorted means
    new_cluster_membership = np.zeros_like(cluster_membership)
    for i, old_cluster in enumerate(sorted_indices):
        new_cluster_membership[cluster_membership == old_cluster] = i
    cluster_membership = new_cluster_membership
    
    # Print results
    print("Mean:")
    for mean in means:
        print(" ".join([f"{x:.3f}" for x in mean]))
    
    print("\nCovariance Matrices:")
    for cov in covs:
        for row in cov:
            print(" ".join([f"{x:.3f}" for x in row]))
        print()
    
    print(f"Iteration count={iterations}")
    
    print("Cluster Membership:")
    for cluster in range(k):
        cluster_points = np.where(cluster_membership == cluster)[0]
        print(",".join(map(str, sorted(cluster_points))))
    
    print("\nSize:", " ".join(map(str, [np.sum(cluster_membership == i) for i in range(k)])))
    
    print(f"Purity: {purity:.3f}")

if __name__ == "__main__":
    main()