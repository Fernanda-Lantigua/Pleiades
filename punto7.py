import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def query_simbad_coords(cluster_name):
    simbad = Simbad()
    result = simbad.query_object(cluster_name)
    if result is not None:
        ra = result['RA'].data[0]
        dec = result['DEC'].data[0]
        coords = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')
        return coords
    else:
        raise ValueError(f"Could not find coordinates for {cluster_name} in Simbad.")

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

# Define old open clusters
clusters = {
    "M67": query_simbad_coords("M67"),
    "NGC 188": query_simbad_coords("NGC 188"),
    "Berkeley 39": query_simbad_coords("Berkeley 39")
}

def process_cluster(name, coords):
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    return results

def apply_clustering(results, algorithm='kmeans', n_clusters=2, eps=0.5, min_samples=5):
    # Handle masked arrays
    ra = np.ma.filled(results['ra'], fill_value=np.nan)
    dec = np.ma.filled(results['dec'], fill_value=np.nan)
    pmra = np.ma.filled(results['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(results['pmdec'], fill_value=np.nan)
    parallax = np.ma.filled(results['parallax'], fill_value=np.nan)
    
    # Stack the data and remove any rows with NaN values
    data = np.vstack((ra, dec, pmra, pmdec, parallax)).T
    valid_data = ~np.isnan(data).any(axis=1)
    data = data[valid_data]
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    labels = model.fit_predict(data_scaled)
    
    # Create a mask for the original data
    cluster_mask = np.zeros(len(results), dtype=bool)
    cluster_mask[valid_data] = (labels == 0)  # Assuming cluster 0 is the main cluster
    
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
    for name, coords in clusters.items():
        print(f"Processing cluster: {name}")
        results = process_cluster(name, coords)
        
        # Apply K-means clustering
        kmeans_mask = apply_clustering(results, algorithm='kmeans')
        
        # Plot results for K-means
        plot_ra_dec(results, kmeans_mask, f"{name}_KMeans")
        plot_pm(results, kmeans_mask, f"{name}_KMeans")
        plot_cmd(results, kmeans_mask, f"{name}_KMeans")
        
        # Apply DBSCAN clustering
        dbscan_mask = apply_clustering(results, algorithm='dbscan')
        
        # Plot results for DBSCAN
        plot_ra_dec(results, dbscan_mask, f"{name}_DBSCAN")
        plot_pm(results, dbscan_mask, f"{name}_DBSCAN")
        plot_cmd(results, dbscan_mask, f"{name}_DBSCAN")

if __name__ == "__main__":
    main()
