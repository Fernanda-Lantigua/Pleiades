from Graph import Graph #importar la función de graficar
from aQuery import aQuery #importar la función que toma datos de Gaia

def main():
    Graph(aQuery()) #Grafica los datos específicos tomados de Gaia

if __name__ == "__main__":
    main()