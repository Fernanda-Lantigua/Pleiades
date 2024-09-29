import numpy as np
import astropy.units as u
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord


def aQuery():
    pleiades_coords = SkyCoord(ra=56.75, dec=24.1167, unit=(u.deg, u.deg), frame='icrs')  # Coordenadas de las Pléyades

    search_radius = 1.5 * u.deg  # Ajusta el radio de búsqueda si es necesario

    # Consulta a Gaia para obtener datos del cúmulo
    job = Gaia.launch_job_async(f"""
    SELECT ra, dec, parallax, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {pleiades_coords.ra.deg}, {pleiades_coords.dec.deg}, {search_radius.to(u.deg).value})
    )=1
    AND parallax BETWEEN 7 AND 9  -- Filtro aproximado para la distancia de Las Pléyades
    """)
    results = job.get_results()

    parallax_arcsec = results['parallax'] / 1000.0  # Convertir a arcsec
    distance_pc = 1 / parallax_arcsec  # Distancia en parsecs

    # Calcular la magnitud absoluta
    results['absolute_magnitude'] = results['phot_g_mean_mag'] - 5 * (np.log10(distance_pc) - 1)

    results['BP-RP'] = results['phot_bp_mean_mag'] - results['phot_rp_mean_mag']

# Mostrar resultados incluyendo el color
    print(results['ra', 'dec', 'parallax', 'phot_g_mean_mag', 'BP-RP', 'absolute_magnitude'])

    return results
