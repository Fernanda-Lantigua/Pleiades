import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord


def aQuery():
    pleiades_coords = SkyCoord(ra=56.75, dec=24.1167, unit=(u.deg, u.deg), frame='icrs')  # Coordenadas de las Pléyades

    search_radius = 1.5 * u.deg  # Ajusta el radio de búsqueda si es necesario

    # Consulta a Gaia para obtener datos del cúmulo incluyendo movimiento propio
    job = Gaia.launch_job_async(f"""
    SELECT ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, pmra, pmdec
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {pleiades_coords.ra.deg}, {pleiades_coords.dec.deg}, {search_radius.to(u.deg).value})
    )=1
    """)

    results = job.get_results()

    # Convertir la paralaje a distancia en parsecs
    parallax_arcsec = results['parallax'] / 1000.0  # Convert to arcsec
    distance_pc = np.where(parallax_arcsec > 0, 1 / parallax_arcsec, np.inf)  # Distance in parsecs

    # Calculate the absolute magnitude, handling potential infinities
    with np.errstate(invalid='ignore'):  # Ignore warnings for invalid values
        results['absolute_magnitude'] = np.where(
            np.isfinite(distance_pc),
            results['phot_g_mean_mag'] - 5 * (np.log10(distance_pc) - 1),
            np.nan
        )  # Distancia en parsecs

    # Calcular la magnitud absoluta
    results['absolute_magnitude'] = results['phot_g_mean_mag'] - 5 * (np.log10(distance_pc) - 1)

    # Calcular el color BP-RP
    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']

    # Mostrar los resultados incluyendo movimiento propio
    f = open("resultados.txt", 'a')

# Escribir encabezado
    f.write("ra, dec, parallax, phot_g_mean_mag, BP-RP, absolute_magnitude, pmra, pmdec\n")

# Escribir datos fila por fila
    for row in results:
        f.write(f"{row['ra']}, {row['dec']}, {row['parallax']}, {row['phot_g_mean_mag']}, "
            f"{row['BP-RP']}, {row['absolute_magnitude']}, {row['pmra']}, {row['pmdec']}\n")

    f.close()

    #print(results['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'BP-RP', 'absolute_magnitude', 'pmra', 'pmdec'])

    return results
