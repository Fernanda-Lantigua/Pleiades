import matplotlib.pyplot as plt
from astropy.table import Table
import numpy as np

def HertzsprungRussell(aTable):
    norm = plt.Normalize(aTable['BP-RP'].min(), aTable['BP-RP'].max())
    colors = plt.cm.coolwarm(norm(aTable['BP-RP']))  # Usar el colormap coolwarm

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(aTable['BP-RP'], aTable['absolute_magnitude'], s=1, color=colors)
    plt.title('Las Pléyades')
    plt.xlabel('Color BP-RP')
    plt.ylabel('Magnitud absoluta')
    plt.colorbar(label='Color BP-RP')  # Agregar barra de color
    plt.gca().invert_yaxis()  # Agregar barra de color
    plt.show()



def ProperMotion(aTable):
    # Ensure we're working with an Astropy Table
    if not isinstance(aTable, Table):
        aTable = Table(aTable)

    # Filter out any rows with invalid proper motion data
    valid_data = ~aTable['pmra'].mask & ~aTable['pmdec'].mask & ~aTable['ra'].mask & ~aTable['dec'].mask & ~aTable['BP-RP'].mask
    
    # Convert columns to numpy arrays, removing masked values
    ra = aTable['ra'][valid_data].data
    dec = aTable['dec'][valid_data].data
    pmra = aTable['pmra'][valid_data].data
    pmdec = aTable['pmdec'][valid_data].data
    bp_rp = aTable['BP-RP'][valid_data].data

    norm = plt.Normalize(np.nanmin(bp_rp), np.nanmax(bp_rp))
    colors = plt.cm.coolwarm(norm(bp_rp))

    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate the scale factor based on data range
    ra_range = np.ptp(ra)
    dec_range = np.ptp(dec)
    pm_scale = 0.1 * min(ra_range, dec_range) / max(np.abs(pmra).max(), np.abs(pmdec).max())

    q = ax.quiver(ra, dec, pmra, pmdec,
                  color=colors, angles='xy', scale_units='xy', 
                  scale=1/pm_scale, width=0.002)

    ax.set_title('Proper Motion of Pleiades')
    ax.set_xlabel('Right Ascension ')
    ax.set_ylabel('Declination ')

    # Add a key for scale
    ax.quiverkey(q, 0.9, 0.95, 10, r'10 mas/yr', labelpos='E', coordinates='figure')

    # Set axis limits to focus on the cluster
    ax.set_xlim(np.nanmin(ra), np.nanmax(ra))
    ax.set_ylim(np.nanmin(dec), np.nanmax(dec))

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Color BP-RP')

    plt.tight_layout()
    plt.show()

    # Print some statistics for debugging
    print(f"Number of stars plotted: {len(ra)}")
    print(f"RA range: {np.nanmin(ra):.2f} to {np.nanmax(ra):.2f}")
    print(f"Dec range: {np.nanmin(dec):.2f} to {np.nanmax(dec):.2f}")
    print(f"pmRA range: {np.nanmin(pmra):.2f} to {np.nanmax(pmra):.2f}")
    print(f"pmDec range: {np.nanmin(pmdec):.2f} to {np.nanmax(pmdec):.2f}")