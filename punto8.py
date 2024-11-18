import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score

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

clusters = {
    
    "NGC 188": SkyCoord(ra=11.806667, dec=85.255278, unit=(u.deg, u.deg), frame='icrs'),
    "Berkeley 39": SkyCoord(ra=116.68750, dec=41.637222, unit=(u.deg, u.deg), frame='icrs')
}

def process_cluster(name, coords):
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    return results

def apply_clustering(results, algorithm='kmeans', n_clusters=2, eps=0.5, min_samples=5):
    pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
    parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
    
    if algorithm in ['kmeans', 'dbscan_parallax_pm']:
        data = np.vstack((pmra, pmdec, parallax)).T
    else:
        ra = np.ma.filled(results['ra'], fill_value=np.nan)
        dec = np.ma.filled(results['dec'], fill_value=np.nan)
        data = np.vstack((ra, dec, pmra, pmdec, parallax)).T
    
    valid_data = ~np.isnan(data).any(axis=1)
    data = data[valid_data]
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'dbscan_parallax_pm':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'hdbscan':
        model = HDBSCAN(min_samples=min_samples)
    
    labels = model.fit_predict(data_scaled)
    
    cluster_mask = np.zeros(len(results), dtype=bool)
    cluster_mask[valid_data] = (labels == 0)
    
    return cluster_mask

def plot_ra_dec(results, members_mask, name):
    plt.figure(figsize=(10, 8))
    plt.scatter(results['ra'][~members_mask], results['dec'][~members_mask], s=1, color='gray', alpha=0.5, label='Non-members')
    plt.scatter(results['ra'][members_mask], results['dec'][members_mask], s=1, color='red', alpha=0.5, label='Members')
    plt.title(f'RA vs Dec for cluster {name}')
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    plt.legend()
    plt.savefig(f"results/RAvsDecCluster_{name}.png")
    plt.close()

def plot_pm(results, members_mask, name):
    plt.figure(figsize=(10, 8))
    plt.scatter(results['pmra'][~members_mask], results['pmdec'][~members_mask], s=10, color='gray', alpha=0.5, label='Non-members')
    plt.scatter(results['pmra'][members_mask], results['pmdec'][members_mask], s=10, color='red', alpha=0.5, label='Members')
    plt.title(f'Proper Motion for cluster {name}')
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('pmDec (mas/yr)')
    plt.legend()
    plt.savefig(f"results/ProperMotionCluster_{name}.png")
    plt.close()

def plot_cmd(results, members_mask, name):
    plt.figure(figsize=(10, 8))
    plt.scatter(results['BP-RP'][~members_mask], results['phot_g_mean_mag'][~members_mask], s=1, color='gray', alpha=0.5, label='Non-members')
    plt.scatter(results['BP-RP'][members_mask], results['phot_g_mean_mag'][members_mask], s=1, color='red', alpha=0.5, label='Members')
    plt.gca().invert_yaxis()
    plt.title(f'CMD of cluster {name}')
    plt.xlabel('Color (BP-RP)')
    plt.ylabel('Apparent Magnitude G')
    plt.legend()
    plt.savefig(f"results/CMDCluster_{name}.png")
    plt.close()

def optimize_kmeans(data_scaled):
    best_n_clusters = None
    best_score = -1
    
    for n_clusters in range(2, 10):
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        if score > best_score:
            best_n_clusters = n_clusters
            best_score = score
    
    return best_n_clusters

def optimize_dbscan(data_scaled):
    best_params = None
    best_score = -1
    
    for eps in np.arange(0.03, 1.0, 0.1):
        for min_samples in range(3, 15):
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data_scaled)
            if len(np.unique(labels)) > 1:
                score = silhouette_score(data_scaled, labels)
                if score > best_score:
                    best_params = (eps, min_samples)
                    best_score = score
    
    return best_params

def optimize_hdbscan(data_scaled):
    best_min_samples = None
    best_score = -1
    
    for min_samples in range(3, 15):
        model = HDBSCAN(min_samples=min_samples)
        labels = model.fit_predict(data_scaled)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(data_scaled, labels)
            if score > best_score:
                best_min_samples = min_samples
                best_score = score
    
    return best_min_samples

def plot_histograms(results, name):
    plt.figure(figsize=(15, 10))
    
    # Histograma de Parallax
    plt.subplot(2, 2, 1)
    plt.hist(results['parallax'], bins=30, color='blue', alpha=0.7)
    plt.title('Parallax Distribution')
    plt.xlabel('Parallax (mas)')
    plt.ylabel('Frequency')
    
    # Histograma de Proper Motion en RA (pmra)
    plt.subplot(2, 2, 2)
    plt.hist(results['pmra'], bins=30, color='green', alpha=0.7)
    plt.title('Proper Motion RA (pmRA)')
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('Frequency')
    
    # Histograma de Proper Motion en Dec (pmdec)
    plt.subplot(2, 2, 3)
    plt.hist(results['pmdec'], bins=30, color='red', alpha=0.7)
    plt.title('Proper Motion Dec (pmDec)')
    plt.xlabel('pmDec (mas/yr)')
    plt.ylabel('Frequency')
    
    # Histograma de BP-RP (Color index)
    plt.subplot(2, 2, 4)
    plt.hist(results['BP-RP'], bins=30, color='purple', alpha=0.7)
    plt.title('BP-RP (Color Index)')
    plt.xlabel('BP-RP')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"results/Histograms_{name}.png")
    plt.close()

def main():
    for name, coords in clusters.items():
        results = process_cluster(name, coords)
        
        # Generar histogramas de las variables clave
        plot_histograms(results, name)
        
        pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
        pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
        parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
        data = np.vstack((pmra, pmdec, parallax)).T
        valid_data = ~np.isnan(data).any(axis=1)
        data = data[valid_data]
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Optimizar K-means
        best_n_clusters = optimize_kmeans(data_scaled)
        print(f"Best k for {name}: {best_n_clusters}")
        
        # Optimizar DBSCAN
        best_dbscan_params = optimize_dbscan(data_scaled)
        print(f"Best DBSCAN params for {name}: eps={best_dbscan_params[0]}, min_samples={best_dbscan_params[1]}")
        
        # Optimizar HDBSCAN
        best_min_samples_hdbscan = optimize_hdbscan(data_scaled)
        print(f"Best HDBSCAN min_samples for {name}: {best_min_samples_hdbscan}")
        
        # Plotear resultados usando los mejores parámetros
        kmeans_mask = apply_clustering(results, algorithm='kmeans', n_clusters=best_n_clusters)
        plot_ra_dec(results, kmeans_mask, f"{name}_KMeans")
        plot_pm(results, kmeans_mask, f"{name}_KMeans")
        plot_cmd(results, kmeans_mask, f"{name}_KMeans")
        
        dbscan_mask = apply_clustering(results, algorithm='dbscan_parallax_pm', eps=best_dbscan_params[0], min_samples=best_dbscan_params[1])
        plot_ra_dec(results, dbscan_mask, f"{name}_DBSCAN")
        plot_pm(results, dbscan_mask, f"{name}_DBSCAN")
        plot_cmd(results, dbscan_mask, f"{name}_DBSCAN")

         # Plotear resultados usando los mejores parámetros para HDBSCAN
        hdbscan_mask = apply_clustering(results, algorithm='hdbscan', min_samples=best_min_samples_hdbscan)
        plot_ra_dec(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_pm(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_cmd(results, hdbscan_mask, f"{name}_HDBSCAN")

if __name__ == '__main__':
    main()
