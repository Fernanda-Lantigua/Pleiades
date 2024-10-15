import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

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
    "M67": SkyCoord(ra=132.825, dec=11.8, unit=(u.deg, u.deg), frame='icrs'),
    "NGC 188": SkyCoord(ra=11.8, dec=85.25, unit=(u.deg, u.deg), frame='icrs'),
    "Berkeley 39": SkyCoord(ra=116.6875, dec=+41.6, unit=(u.deg, u.deg), frame='icrs')
}

def process_cluster(name, coords):
    print(f"Processing cluster: {name}")
    results = query_cluster_stars(coords, search_radius=0.5*u.deg)
    
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']
    
    pmra_mean = np.mean(results['pmra'])
    pmdec_mean = np.mean(results['pmdec'])
    pmra_std = np.std(results['pmra'])
    pmdec_std = np.std(results['pmdec'])
    
    members = results[
        (np.abs(results['pmra'] - pmra_mean) < 2 * pmra_std) &
        (np.abs(results['pmdec'] - pmdec_mean) < 2 * pmdec_std)
    ]
    
    mask = ~members['ra'].mask & ~members['dec'].mask & ~members['pmra'].mask & ~members['pmdec'].mask
    members_clean = members[mask]
    
    return members_clean

def plot_proper_motion(members_clean, name):
    ra = members_clean['ra'].data
    dec = members_clean['dec'].data
    pmra = members_clean['pmra'].data
    pmdec = members_clean['pmdec'].data
    
    pm_scale = np.sqrt(np.median(pmra**2 + pmdec**2))
    
    plt.figure(figsize=(10, 8))
    plt.scatter(ra, dec, s=1, color='gray', alpha=0.5)
    q = plt.quiver(ra, dec, pmra, pmdec, 
                   angles='xy', scale_units='xy', 
                   scale=pm_scale*100,
                   width=0.001, color='red', alpha=0.7)
    plt.quiverkey(q, 0.9, 0.95, pm_scale, f'{pm_scale:.1f} mas/yr', 
                  labelpos='E', coordinates='figure')
    plt.title(f'Proper motion of stars in cluster {name}')
    plt.xlabel('Right Ascension')
    plt.ylabel('Declination')
    plt.xlim(ra.min(), ra.max())
    plt.ylim(dec.min(), dec.max())
    plt.savefig("results/ProperMotionof{name}.png".format(name=name))
    
    print(f"Number of stars: {len(ra)}")
    print(f"RA range: {ra.min():.2f} to {ra.max():.2f}")
    print(f"Dec range: {dec.min():.2f} to {dec.max():.2f}")
    print(f"pmRA range: {pmra.min():.2f} to {pmra.max():.2f}")
    print(f"pmDec range: {pmdec.min():.2f} to {pmdec.max():.2f}")

def plot_pmra_pmdec(members_clean, name):
    pmra = members_clean['pmra'].data
    pmdec = members_clean['pmdec'].data
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pmra, pmdec, s=1, alpha=0.5)
    plt.title(f'Proper Motion Distribution of {name}')
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('pmDec (mas/yr)')
    plt.savefig("results/ProperMotionDistributionof{name}.png".format(name=name))

def plot_cmd(members_clean, name):
    norm = plt.Normalize(members_clean['BP-RP'].min(), members_clean['BP-RP'].max())
    colors = plt.cm.viridis(norm(members_clean['BP-RP']))
    plt.figure(figsize=(8, 6))
    plt.scatter(members_clean['BP-RP'], members_clean['phot_g_mean_mag'], s=1, color=colors)
    plt.gca().invert_yaxis()
    plt.title(f'CMD of cluster {name}')
    plt.xlabel('Color BP-RP')
    plt.ylabel('Magnitude G')
    plt.savefig("results/CMDCluster{name}.png".format(name=name))

def apply_clustering(members_clean, n_clusters=2):
    ra = np.ma.filled(members_clean['ra'], fill_value=np.nan)
    dec = np.ma.filled(members_clean['dec'], fill_value=np.nan)
    pmra = np.ma.filled(members_clean['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(members_clean['pmdec'], fill_value=np.nan)
    
    mask = ~np.isnan(ra) & ~np.isnan(dec) & ~np.isnan(pmra) & ~np.isnan(pmdec)
    data = np.vstack((ra[mask], dec[mask], pmra[mask], pmdec[mask])).T
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_scaled)
    
    return labels, mask

def apply_dbscan(members_clean, eps=0.5, min_samples=5):
    ra = np.ma.filled(members_clean['ra'], fill_value=np.nan)
    dec = np.ma.filled(members_clean['dec'], fill_value=np.nan)
    pmra = np.ma.filled(members_clean['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(members_clean['pmdec'], fill_value=np.nan)
    
    mask = ~np.isnan(ra) & ~np.isnan(dec) & ~np.isnan(pmra) & ~np.isnan(pmdec)
    data = np.vstack((ra[mask], dec[mask], pmra[mask], pmdec[mask])).T
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(data_scaled)
    
    return labels, mask

def apply_gmm(members_clean, n_components=2):
    ra = np.ma.filled(members_clean['ra'], fill_value=np.nan)
    dec = np.ma.filled(members_clean['dec'], fill_value=np.nan)
    pmra = np.ma.filled(members_clean['pmra'], fill_value=np.nan)
    pmdec = np.ma.filled(members_clean['pmdec'], fill_value=np.nan)
    
    mask = ~np.isnan(ra) & ~np.isnan(dec) & ~np.isnan(pmra) & ~np.isnan(pmdec)
    data = np.vstack((ra[mask], dec[mask], pmra[mask], pmdec[mask])).T
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    gmm = GaussianMixture(n_components=n_components)
    labels = gmm.fit_predict(data_scaled)
    
    return labels, mask

def plot_clusters_proper_motion(members_clean, labels, mask, name):
    pmra = members_clean['pmra'].data[mask]
    pmdec = members_clean['pmdec'].data[mask]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(pmra, pmdec, c=labels, cmap='viridis', s=10)
    plt.title(f'Proper Motion Clustering of {name}')
    plt.xlabel('pmRA (mas/yr)')
    plt.ylabel('pmDec (mas/yr)')
    plt.colorbar(label='Cluster Label')
    plt.grid(True)
    plt.savefig("results/ProperMotionClusteringof{name}.png".format(name=name))

def main():
    # Process and plot each cluster
    for name, coords in clusters.items():
        print(f"Processing cluster: {name}")
        members_clean = process_cluster(name, coords)
        
        # Step 1: Plot the proper motion data
        plot_proper_motion(members_clean, name)
        
        # New Step: Plot pmra vs pmdec scatter plot
        plot_pmra_pmdec(members_clean, name)
        
        # Step 2: Plot the CMD
        plot_cmd(members_clean, name)
        
        # Step 3: Apply clustering algorithms and plot results
        # K-means
        kmeans_labels, kmeans_mask = apply_clustering(members_clean, n_clusters=3)
        plot_clusters_proper_motion(members_clean, kmeans_labels, kmeans_mask, f"{name} (K-means)")
        
        # DBSCAN
        dbscan_labels, dbscan_mask = apply_dbscan(members_clean)
        plot_clusters_proper_motion(members_clean, dbscan_labels, dbscan_mask, f"{name} (DBSCAN)")
        
        # Gaussian Mixture Model
        gmm_labels, gmm_mask = apply_gmm(members_clean)
        plot_clusters_proper_motion(members_clean, gmm_labels, gmm_mask, f"{name} (GMM)")

if __name__ == "__main__":
    main()