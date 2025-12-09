import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

def estimate_reach(data, intrinsic_dim, k_neighbors):
    """
    Estimates the reach of a manifold from a point cloud.
    
    Parameters:
    - data: numpy array of shape (n_samples, ambient_dim)
    - intrinsic_dim: (int) The dimension 'd' of the manifold M
    - k_neighbors: (int) Number of neighbors to use for tangent space estimation
    
    Returns:
    - tau_hat: The estimated reach
    """
    n_samples = data.shape[0]
    
    # --- Step 1: Estimate Tangent Spaces (Section 6) ---
    # We use Local PCA: The tangent space at x is approximated by the 
    # principal components of its k-nearest neighbors.
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(data)
    _, indices = nbrs.kneighbors(data)
    
    tangent_bases = []
    
    for i in range(n_samples):
        # Get neighborhood of point i
        neighbor_points = data[indices[i]]
        
        # Center the neighborhood
        neighbor_mean = np.mean(neighbor_points, axis=0)
        centered_neighbors = neighbor_points - neighbor_mean
        
        # PCA to find the first 'd' principal components (tangent basis)
        pca = PCA(n_components=intrinsic_dim)
        pca.fit(centered_neighbors)
        
        # The components_ attribute contains the basis vectors of the tangent space
        tangent_bases.append(pca.components_) 
    
    tangent_bases = np.array(tangent_bases)
    
    # --- Step 2: Compute Reach Estimator (Equation 6.1) ---
    # Formula: min ||y - x||^2 / (2 * dist(y - x, T_x))
    
    min_ratio = np.inf
    
    # Naive implementation is O(N^2). For large N, one might restrict 
    # the search to local neighborhoods or use spatial indexing, 
    # though the paper implies a global infimum.
    for i in range(n_samples):
        x = data[i]
        Tx = tangent_bases[i]  # Basis vectors for tangent space at x (shape: d x D)
        
        for j in range(n_samples):
            if i == j:
                continue
            
            y = data[j]
            v = y - x # Vector from x to y
            norm_v_sq = np.sum(v**2)
            
            # Calculate distance of v to the tangent space Tx
            # Project v onto Tx: proj_v = sum((v . basis_k) * basis_k)
            # Since Tx basis is orthonormal from PCA:
            coeffs = np.dot(Tx, v) # shape (d,)
            proj_v = np.dot(coeffs, Tx) # shape (D,)
            
            # The perpendicular component (distance to space)
            perp_v = v - proj_v
            dist_to_Tx = np.linalg.norm(perp_v)
            
            # Avoid division by zero if point lies exactly on tangent space
            if dist_to_Tx < 1e-12:
                continue
                
            # The paper's ratio
            ratio = norm_v_sq / (2 * dist_to_Tx)
            
            if ratio < min_ratio:
                min_ratio = ratio

    return min_ratio

# --- Example Usage ---
if __name__ == "__main__":
    # Create a synthetic circle in 2D (Reach = Radius = 1.0)
    theta = np.linspace(0, 2*np.pi, 1000, endpoint=False)
    circle_x = np.cos(theta)
    circle_y = np.sin(theta)
    X = np.column_stack([circle_x, circle_y])
    
    # Intrinsic dimension of a circle is 1
    # We use small k for local tangent estimation
    estimated_reach = estimate_reach(X, intrinsic_dim=1, k_neighbors=9)
    
    print(f"True Reach: 1.0")
    print(f"Estimated Reach: {estimated_reach:.4f}")