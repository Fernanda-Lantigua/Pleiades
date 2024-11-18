import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score

# Función para realizar la consulta a Gaia
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

# Definir coordenadas de los cúmulos
clusters = {
    "M67": SkyCoord(ra=132.825, dec=11.814722, unit=(u.deg, u.deg), frame='icrs'),
    "NGC 188": SkyCoord(ra=11.806667, dec=85.255278, unit=(u.deg, u.deg), frame='icrs'),
    "Berkeley 39": SkyCoord(ra=116.68750, dec=41.637222, unit=(u.deg, u.deg), frame='icrs')
}

# Procesar cada cúmulo y calcular los colores
def process_cluster(name, coords):
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    return results

# Aplicar clustering utilizando KMeans, DBSCAN o HDBSCAN
def apply_clustering(results, algorithm='kmeans', n_clusters=2, eps=0.5, min_samples=5):
    pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
    parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
    
    data = np.vstack((pmra, pmdec, parallax)).T
    valid_data = ~np.isnan(data).any(axis=1)
    data = data[valid_data]
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Seleccionar el modelo de clustering
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'hdbscan':
        model = HDBSCAN(min_samples=min_samples)
    
    labels = model.fit_predict(data_scaled)
    
    # Contar el número de clústeres (excluyendo ruido etiquetado como -1)
    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters detected by {algorithm}: {n_clusters}")

    cluster_mask = np.zeros(len(results), dtype=bool)
    cluster_mask[valid_data] = (labels != -1)
    
    return cluster_mask, labels, n_clusters

# Funciones para graficar
def plot_ra_dec(results, members_mask, name):
    plt.figure(figsize=(10, 8))
    plt.scatter(results['ra'][~members_mask], results['dec'][~members_mask], s=1, color='gray', alpha=0.5, label='Non-members')
    plt.scatter(results['ra'][members_mask], results['dec'][members_mask], s=1, color='red', alpha=0.5, label='Members')
    plt.title(f'RA vs Dec for cluster {name}')
    plt.xlabel('Right Ascension (deg)')
    plt.ylabel('Declination (deg)')
    plt.legend()
    plt.savefig(f"results/RAvsDec_{name}.png")
    plt.close()

def plot_pm(results, members_mask, name):
    plt.figure(figsize=(10, 8))
    plt.scatter(results['pmra'][~members_mask], results['pmdec'][~members_mask], s=10, color='gray', alpha=0.5, label='Non-members')
    plt.scatter(results['pmra'][members_mask], results['pmdec'][members_mask], s=10, color='red', alpha=0.5, label='Members')
    plt.title(f'Proper Motion for cluster {name}')
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('pmDec (mas/yr)')
    plt.legend()
    plt.savefig(f"results/ProperMotion_{name}.png")
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
    plt.savefig(f"results/CMD_{name}.png")
    plt.close()

# Función principal para ejecutar el análisis
def main():
    for name, coords in clusters.items():
        results = process_cluster(name, coords)
        
        # Aplicar KMeans
        kmeans_mask, kmeans_labels, kmeans_clusters = apply_clustering(results, algorithm='kmeans', n_clusters=4)
        print(f"KMeans detected {kmeans_clusters} clusters for {name}.")
        plot_ra_dec(results, kmeans_mask, f"{name}_KMeans")
        plot_pm(results, kmeans_mask, f"{name}_KMeans")
        plot_cmd(results, kmeans_mask, f"{name}_KMeans")
        
        # Aplicar DBSCAN
        dbscan_mask, dbscan_labels, dbscan_clusters = apply_clustering(results, algorithm='dbscan', eps=0.3, min_samples=10)
        print(f"DBSCAN detected {dbscan_clusters} clusters for {name}.")
        plot_ra_dec(results, dbscan_mask, f"{name}_DBSCAN")
        plot_pm(results, dbscan_mask, f"{name}_DBSCAN")
        plot_cmd(results, dbscan_mask, f"{name}_DBSCAN")

        # Aplicar HDBSCAN
        hdbscan_mask, hdbscan_labels, hdbscan_clusters = apply_clustering(results, algorithm='hdbscan', min_samples=15)
        print(f"HDBSCAN detected {hdbscan_clusters} clusters for {name}.")
        plot_ra_dec(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_pm(results, hdbscan_mask, f"{name}_HDBSCAN")
        plot_cmd(results, hdbscan_mask, f"{name}_HDBSCAN")

if __name__ == '__main__':
    main()
