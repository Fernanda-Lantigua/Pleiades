from Graph import HertzsprungRussell, ProperMotion #importar la función de graficar
from aQuery import aQuery #importar la función que toma datos de Gaia

def main():
    aTable = aQuery()
    HertzsprungRussell(aTable) #Grafica los datos específicos tomados de Gaia
    ProperMotion(aTable)

if __name__ == "__main__":
    main()