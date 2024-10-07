import numpy as np
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

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

# Definir los cúmulos abiertos viejos
clusters = {
    "M67": SkyCoord(ra=132.825, dec=11.8, unit=(u.deg, u.deg), frame='icrs'),
    "NGC 188": SkyCoord(ra=11.8, dec=85.25, unit=(u.deg, u.deg), frame='icrs'),
    "Berkeley 39": SkyCoord(ra=116.6875, dec=+41.6, unit=(u.deg, u.deg), frame='icrs')
}

def process_cluster(name, coords):
    print(f"Procesando el cúmulo: {name}")
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
    plt.title(f'Movimiento propio de las estrellas del cúmulo {name}')
    plt.xlabel('Ascensión Recta ')
    plt.ylabel('Declinación ')
    plt.xlim(ra.min(), ra.max())
    plt.ylim(dec.min(), dec.max())
    plt.show()
    
    print(f"Número de estrellas: {len(ra)}")
    print(f"Rango RA: {ra.min():.2f} a {ra.max():.2f}")
    print(f"Rango Dec: {dec.min():.2f} a {dec.max():.2f}")
    print(f"Rango pmRA: {pmra.min():.2f} a {pmra.max():.2f}")
    print(f"Rango pmDec: {pmdec.min():.2f} a {pmdec.max():.2f}")

def plot_cmd(members_clean, name):
    norm = plt.Normalize(members_clean['BP-RP'].min(), members_clean['BP-RP'].max())
    colors = plt.cm.viridis(norm(members_clean['BP-RP']))
    plt.figure(figsize=(8, 6))
    plt.scatter(members_clean['BP-RP'], members_clean['phot_g_mean_mag'], s=1, color=colors)
    plt.gca().invert_yaxis()
    plt.title(f'CMD del cúmulo {name}')
    plt.xlabel('Color BP-RP')
    plt.ylabel('Magnitud G')
    plt.show()

# Procesar y graficar cada cúmulo
for name, coords in clusters.items():
    members_clean = process_cluster(name, coords)
    plot_proper_motion(members_clean, name)
    plot_cmd(members_clean, name)