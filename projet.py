import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

df = pd.read_csv('dataset.txt', sep='\t')

# Determine the number of columns in the DataFrame
num_columns = df.shape[1]

# # Create a figure with a subplot for each column
# fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(num_columns * 6, 6))

# # Create individual box plots for each column
# for i, column in enumerate(df.columns):
#     df.boxplot(column=column, ax=axes[i], vert=False)
#     axes[i].set_title(f'Box Plot of {column}')
#     axes[i].set_xlabel('Values')

# plt.tight_layout()
# plt.show()

transformer = preprocessing.StandardScaler().fit(df)
df_transformed = transformer.transform(df)
#print(df_transformed)

df = pd.DataFrame(df_transformed, columns = ['AT', 'V', 'AP', 'RH', 'EP'])

# var = df['EP'].var()
# print(var)

# Création de l'objet PCA avec 2 composantes principales
pca = PCA(n_components=2)

# Application de l'ACP sur les données standardisées
pca.fit(df)

# Affichage des vecteurs propres des deux premières composantes principales
print("Vecteur propre de la première composante principale:", pca.components_[0])
print("Vecteur propre de la deuxième composante principale:", pca.components_[1])


# Effectuer l'ACP
scores = pca.transform(df)

# Créer un biplot
plt.figure(figsize=(10, 8))

# Tracer les scores
plt.scatter(scores[:, 0], scores[:, 1], c='gray', alpha=0.5)

# Tracer les vecteurs propres
for i, col_name in enumerate(df.columns):
    plt.arrow(0, 0, pca.components_[0, i] * max(scores[:, 0]), pca.components_[1, i] * max(scores[:, 1]), 
              color='red', width=0.005, head_width=0.05)
    plt.text(pca.components_[0, i] * max(scores[:, 0]) * 1.2, pca.components_[1, i] * max(scores[:, 1]) * 1.2, 
             col_name, color='red')

# Ajouter des labels et un titre
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot des deux premières composantes principales")

# Afficher le graphique
plt.show()
