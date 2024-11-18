import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.interpolate import griddata

# Define functions and constants
clusters = {
    "M67": SkyCoord(ra=132.825, dec=11.814722, unit=(u.deg, u.deg), frame='icrs'),
    "NGC 188": SkyCoord(ra=11.806667, dec=85.255278, unit=(u.deg, u.deg), frame='icrs'),
    "Berkeley 39": SkyCoord(ra=116.68750, dec=41.637222, unit=(u.deg, u.deg), frame='icrs')
}

def query_cluster_stars(cluster_coords, search_radius):
    job = Gaia.launch_job_async(f"""
    SELECT ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {cluster_coords.ra.deg}, {cluster_coords.dec.deg}, {search_radius.to(u.deg).value})
    )=1
    """)
    return job.get_results()

def process_cluster(name, coords):
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    return results

def apply_clustering(results, algorithm='kmeans', n_clusters=2, eps=0.5, min_samples=5):
    pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
    parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
    data = np.vstack((pmra, pmdec, parallax)).T
    valid_data = ~np.isnan(data).any(axis=1)
    data = data[valid_data]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'dbscan' or algorithm == 'dbscan_parallax_pm':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'hdbscan':
        model = HDBSCAN(min_samples=min_samples)
    
    labels = model.fit_predict(data_scaled)
    cluster_mask = np.zeros(len(results), dtype=bool)
    cluster_mask[valid_data] = (labels == 0)
    return cluster_mask

# Optimization functions with plotting for Figure 2-style
def optimize_kmeans(data_scaled):
    scores = []
    for n_clusters in range(2, 10):
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        scores.append((n_clusters, score))
    plot_optimization_scores(scores, "K-means", "n_clusters")
    best_n_clusters = max(scores, key=lambda x: x[1])[0]
    return best_n_clusters

def optimize_dbscan(data_scaled):
    scores = []
    for eps in np.arange(0.03, 1.0, 0.1):
        for min_samples in range(3, 15):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data_scaled)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data_scaled, labels)
                scores.append((eps, min_samples, score))
    
    plot_optimization_scores_3d(scores, "DBSCAN")
    best_params = max(scores, key=lambda x: x[2])[:2]
    return best_params

def plot_optimization_scores_3d(scores, algorithm_name):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Extract the individual parameters and silhouette scores
    eps_vals = np.array([score[0] for score in scores])
    min_samples_vals = np.array([score[1] for score in scores])
    silhouette_scores = np.array([score[2] for score in scores])

    # Create a grid over the eps and min_samples values
    eps_grid, min_samples_grid = np.meshgrid(
        np.linspace(eps_vals.min(), eps_vals.max(), 100),
        np.linspace(min_samples_vals.min(), min_samples_vals.max(), 100)
    )

    # Interpolate the silhouette scores to fill the grid
    silhouette_grid = griddata(
        (eps_vals, min_samples_vals), silhouette_scores, (eps_grid, min_samples_grid), method='cubic'
    )

    # Plot the surface
    surf = ax.plot_surface(eps_grid, min_samples_grid, silhouette_grid, cmap='viridis', edgecolor='none')
    ax.set_title(f"{algorithm_name} Optimization: Silhouette Score in 3D")
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")
    ax.set_zlabel("Silhouette Score")

    # Add a color bar to indicate the magnitude of the silhouette scores
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

    # Save the figure
    plt.savefig(f"Results/{algorithm_name}_Optimization.png")
    plt.close()

def optimize_hdbscan(data_scaled):
    scores = []
    for min_samples in range(3, 15):
        model = HDBSCAN(min_samples=min_samples)
        labels = model.fit_predict(data_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data_scaled, labels)
            scores.append((min_samples, score))
    plot_optimization_scores(scores, "HDBSCAN", "min_samples")
    best_min_samples = max(scores, key=lambda x: x[1])[0]
    return best_min_samples

# Plotting function for optimization scores (Figure 2-style)
def plot_optimization_scores(scores, algorithm_name, param_label):
    x_vals = [x[0] for x in scores]
    y_vals = [x[1] for x in scores]
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, marker='o')
    plt.title(f"{algorithm_name} Optimization: Score vs {param_label}")
    plt.xlabel(param_label)
    plt.ylabel("Silhouette Score")
    plt.savefig(f"Results/{algorithm_name}_Optimization.png")
    plt.close()

# Figure 14-style histogram
def plot_membership_histogram(results, dbscan_mask, hdbscan_mask, kmeans_mask, name):
    plt.figure(figsize=(10, 6))
    plt.hist(results['parallax'], bins=30, color='lightgray', alpha=0.5, label='All Gaia Data')
    plt.hist(results['parallax'][dbscan_mask], bins=30, color='blue', alpha=0.5, label='DBSCAN Members')
    plt.hist(results['parallax'][hdbscan_mask], bins=30, color='orange', alpha=0.5, label='HDBSCAN Members')
    plt.hist(results['parallax'][kmeans_mask], bins=30, color='green', alpha=0.5, label='K-means Members')
    plt.xlabel('Parallax (mas)')
    plt.ylabel('Frequency')
    plt.title(f'Membership Histogram for {name}')
    plt.legend()
    plt.savefig(f"Results/MembershipHistogram_{name}.png")
    plt.close()

# Main function
def main():
    for name, coords in clusters.items():
        results = process_cluster(name, coords)
        pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
        pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
        parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
        data = np.vstack((pmra, pmdec, parallax)).T
        valid_data = ~np.isnan(data).any(axis=1)
        data = data[valid_data]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        best_n_clusters = optimize_kmeans(data_scaled)
        best_dbscan_params = optimize_dbscan(data_scaled)
        best_min_samples_hdbscan = optimize_hdbscan(data_scaled)
        
        kmeans_mask = apply_clustering(results, 'kmeans', n_clusters=best_n_clusters)
        dbscan_mask = apply_clustering(results, 'dbscan_parallax_pm', eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
        hdbscan_mask = apply_clustering(results, 'hdbscan', min_samples=best_min_samples_hdbscan)

        plot_membership_histogram(results, dbscan_mask, hdbscan_mask, kmeans_mask, name)

if __name__ == '__main__':
    main()
