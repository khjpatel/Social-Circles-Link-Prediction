import numpy as np
from scipy.stats import multivariate_normal

# Load dataset
filename = 'iris.data'
data = np.loadtxt(filename, delimiter=",", usecols=[0, 1, 2, 3])
labels = np.loadtxt(filename, delimiter=",", usecols=[4], dtype=str)

# Initialize parameters
def initialize_parameters(data, k):
    n, d = data.shape
    means = []
    covariances = []
    for i in range(k):
        cluster_data = data[i * (n // k):(i + 1) * (n // k)]
        means.append(np.mean(cluster_data, axis=0))
        covariances.append(np.cov(cluster_data.T))
    return np.array(means), np.array(covariances)

k = 3
means, covariances = initialize_parameters(data, k)

# E-step
def e_step(data, means, covariances, weights):
    responsibilities = np.zeros((len(data), len(means)))
    for i in range(len(means)):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        responsibilities[:, i] = weights[i] * rv.pdf(data)
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# M-step
def m_step(data, responsibilities):
    n_k = responsibilities.sum(axis=0)
    weights = n_k / len(data)
    means = np.dot(responsibilities.T, data) / n_k[:, None]
    covariances = []
    for i in range(len(means)):
        diff = data - means[i]
        cov = np.dot((responsibilities[:, i] * diff.T), diff) / n_k[i]
        covariances.append(cov)
    return np.array(means), np.array(covariances), weights

# Convergence check
def check_convergence(old_means, new_means, threshold=1e-6):
    distances = np.linalg.norm(new_means - old_means, axis=1)
    return np.sum(distances) < threshold

# Run EM algorithm
weights = np.ones(k) / k
max_iterations = 100
threshold = 1e-6
for iteration in range(max_iterations):
    responsibilities = e_step(data, means, covariances, weights)
    new_means, new_covariances, weights = m_step(data, responsibilities)
    if check_convergence(means, new_means, threshold):
        break
    means, covariances = new_means, new_covariances

print("Mean:", means)
print("Covariance Matrices:", covariances)
print("Iteration count:", iteration + 1)

# Cluster assignments and purity score
cluster_assignments = np.argmax(responsibilities, axis=1)

def purity_score(assignments, true_labels, k):
    purity = 0
    for cluster in range(k):
        cluster_points = true_labels[assignments == cluster]
        max_label_count = max([np.sum(cluster_points == label) for label in set(true_labels)])
        purity += max_label_count
    return purity / len(true_labels)

true_labels = np.array([label for label in labels])
purity = purity_score(cluster_assignments, true_labels, k)

print("Cluster Membership:", cluster_assignments)
print("Size:", [np.sum(cluster_assignments == i) for i in range(k)])
print("Purity:", round(purity, 3))

# Save results
def save_results(means, covariances, iteration, cluster_assignments, purity, k):
    with open("lastname-assign3.txt", "w") as f:
        f.write(f"Mean: {means}\n")
        f.write(f"Covariance Matrices: {covariances}\n")
        f.write(f"Iteration count: {iteration + 1}\n")
        f.write(f"Cluster Membership: {cluster_assignments}\n")
        f.write(f"Size: {[np.sum(cluster_assignments == i) for i in range(k)]}\n")
        f.write(f"Purity: {round(purity, 3)}\n")

save_results(means, covariances, iteration, cluster_assignments, purity, k)
print("Results saved to lastname-assign3.txt")
