import matplotlib.pyplot as plt

def Graph(aTable):
    norm = plt.Normalize(aTable['BP-RP'].min(), aTable['BP-RP'].max())
    colors = plt.cm.coolwarm(norm(aTable['BP-RP']))  # Usar el colormap coolwarm

    # Graficar
    plt.figure(figsize=(8, 6))
    plt.scatter(aTable['BP-RP'], aTable['absolute_magnitude'], s=1, color=colors)
    plt.title('Las pleiades')
    plt.xlabel('Color BP-RP')
    plt.ylabel('Magnitud absoluta')
    plt.colorbar(label='Color BP-RP')  # Agregar barra de color
    plt.show()