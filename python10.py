import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN  # Importar HDBSCAN

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

# Definir los cúmulos abiertos con coordenadas actualizadas y parámetros específicos para clustering
clusters = {
    "M67": {
        "coords": SkyCoord(ra=132.825, dec=11.814722, unit=(u.deg, u.deg), frame='icrs'),
        "eps": 0.0008,
        "min_samples": 3
    },
    "NGC 188": {
        "coords": SkyCoord(ra=11.806667, dec=85.255278, unit=(u.deg, u.deg), frame='icrs'),
        "eps": 0.0009,
        "min_samples": 3
    },
    "Berkeley 39": {
        "coords": SkyCoord(ra=116.68750, dec=41.637222, unit=(u.deg, u.deg), frame='icrs'),
        "eps": 0.0006,
        "min_samples": 3
    }
}

def process_cluster(name, cluster_info):
    coords = cluster_info['coords']
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    return results

def apply_clustering(results, algorithm='kmeans', n_clusters=2, eps=0.5, min_samples=5):
    # Manejo de arrays con máscara
    pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
    parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
    
    # Usar solo pmra, pmdec y parallax para K-means y DBSCAN con proper motion y parallax
    if algorithm in ['kmeans', 'dbscan_parallax_pm']:
        data = np.vstack((pmra, pmdec, parallax)).T
    else:
        # Usar todas las variables para otros algoritmos
        ra = np.ma.filled(results['ra'], fill_value=np.nan)
        dec = np.ma.filled(results['dec'], fill_value=np.nan)
        data = np.vstack((ra, dec, pmra, pmdec, parallax)).T
    
    # Filtrar datos válidos
    valid_data = ~np.isnan(data).any(axis=1)
    data = data[valid_data]
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'dbscan_parallax_pm':  # DBSCAN solo con pmra, pmdec, parallax
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'hdbscan':  # Implementar HDBSCAN
        model = HDBSCAN(min_samples=min_samples)
    
    labels = model.fit_predict(data_scaled)
    
    # Crear máscara para los datos originales
    cluster_mask = np.zeros(len(results), dtype=bool)
    cluster_mask[valid_data] = (labels == 0)  # Asumimos que el cluster 0 es el principal
    
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

def main():
    for name, cluster_info in clusters.items():
        print(f"Processing cluster: {name}")
        results = process_cluster(name, cluster_info)
        
        # Aplicar K-means con proper motion y parallax
        kmeans_mask = apply_clustering(results, algorithm='kmeans')
        plot_ra_dec(results, kmeans_mask, f"{name}_KMeans")
        plot_pm(results, kmeans_mask, f"{name}_KMeans")
        plot_cmd(results, kmeans_mask, f"{name}_KMeans")
        
        # Aplicar DBSCAN variando eps y usando valores específicos de eps y min_samples
        eps_value = cluster_info['eps']
        min_samples_value = cluster_info['min_samples']
        dbscan_mask = apply_clustering(results, algorithm='dbscan_parallax_pm', eps=eps_value, min_samples=min_samples_value)
        plot_ra_dec(results, dbscan_mask, f"{name}_DBSCAN")
        plot_pm(results, dbscan_mask, f"{name}_DBSCAN")
        plot_cmd(results, dbscan_mask, f"{name}_DBSCAN")
        
        # Aplicar HDBSCAN
        hdbscan_mask = apply_clustering(results, algorithm='hdbscan', min_samples=min_samples_value)
        plot_ra_dec(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_pm(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_cmd(results, hdbscan_mask, f"{name}_HDBSCAN")

if __name__ == "__main__":
    main()