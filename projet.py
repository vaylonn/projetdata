import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t
import scipy.stats as stats

df = pd.read_csv('dataset.txt', sep='\t')

# Determine the number of columns in the DataFrame
num_columns = df.shape[1]
    
# Create a figure with a subplot for each column
fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(num_columns * 6, 6))

# Create individual box plots for each column
for i, column in enumerate(df.columns):
    df.boxplot(column=column, ax=axes[i], vert=False)
    axes[i].set_title(f'Box Plot of {column}')
    axes[i].set_xlabel('Values')

plt.tight_layout()
plt.show()

transformer = preprocessing.StandardScaler().fit(df)
df_transformed = transformer.transform(df)
#print(df_transformed)

df = pd.DataFrame(df_transformed, columns = ['AT', 'V', 'AP', 'RH', 'EP'])

# var = df['EP'].var()
# print(var)

# Création de l'objet PCA avec 2 composantes principales
pca = PCA(n_components=3)

# Application de l'ACP sur les données standardisées
pca.fit(df)

# Affichage des vecteurs propres des deux premières composantes principales
print("Vecteur propre de la première composante principale:", pca.components_[0])
print("Vecteur propre de la deuxième composante principale:", pca.components_[1])
print("Vecteur propre de la première composante principale:", pca.components_[2])

# Calcul du PVE par chaque composante
pve = pca.explained_variance_ratio_
print("Pourcentage de variance expliquée par chaque composante :", pve*100)

# Calcul du PVE cumulatif
pve_cumulative = np.cumsum(pve)
print("PVE cumulatif :", pve_cumulative*100)

# Nombre de composantes
n_components = len(pve)

# Création du plot du PVE par composante
plt.figure(figsize=(10, 3))

# Barplot du PVE par chaque composante
plt.bar(range(1, n_components + 1), pve*100, alpha=0.5, align='center',
        label='PVE individuel')

# Ligne du PVE cumulatif
plt.step(range(1, n_components + 1), pve_cumulative*100, where='mid',
         label='PVE cumulatif')

plt.ylabel('Pourcentage de Variance Expliquée')
plt.xlabel('Composantes principales')
plt.legend(loc='best')
plt.title('PVE et PVE cumulatif par composante')

plt.show()

# Décider du nombre de composantes à conserver
# Cela dépendra de votre critère spécifique, par exemple, un seuil de 80-90% pour le PVE cumulatif.

# Effectuer l'ACP
scores = pca.transform(df)

# Créer un biplot
plt.figure(figsize=(10, 6))

plt.scatter(scores[:, 0], scores[:, 1], c=df['V'], alpha=0.5, cmap='viridis')

# Ajouter une barre de couleur pour interpréter les couleurs
plt.colorbar(label="Vide d'échappement (V)")
# Tracer les vecteurs propres (loadings)
for i, col_name in enumerate(df.columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], 
              color='red', width=0.005, head_width=0.05)
    plt.text(pca.components_[0, i], pca.components_[1, i], 
             col_name, color='red')

# Ajouter un cercle de corrélation
circle = plt.Circle((0, 0), 1, color='blue', fill=False)
plt.gca().add_artist(circle)

# Définir les limites du graphique
plt.xlim(-1, 1)
plt.ylim(-1, 1)

# Assurer que les axes sont de même échelle (éviter la déformation du cercle en ellipse)
plt.axis('equal')

# Ajouter des labels et un titre
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Biplot avec cercle de corrélation des deux premières composantes principales")

# Afficher le graphique
plt.show()



# Charger le dataset
# Calculer la matrice de corrélation
correlation_matrix = df.corr()
print(correlation_matrix)

# Trouver la variable la plus corrélée avec 'EP'
ep_correlations = correlation_matrix['EP'].drop('EP')  # Exclure la corrélation de 'EP' avec elle-même
most_correlated_variable = ep_correlations.abs().idxmax()
most_correlated_value = ep_correlations[most_correlated_variable]

print(f"La variable la plus corrélée avec 'EP' est {most_correlated_variable} avec un coefficient de {most_correlated_value:.2f}.")

# Préparer les données pour la régression
X = df[['AT']]  # Feature variable 'AT' dans un DataFrame
y = df['EP']    # Target variable 'EP'

# Initialiser et ajuster le modèle de régression linéaire
model = LinearRegression()
model.fit(X, y)

# Obtenir les estimations des coefficients
beta_0 = model.intercept_
beta_1 = model.coef_[0]

print(f"Estimation du coefficient (intercept) β0: {beta_0:.2f}")
print(f"Estimation du coefficient pour 'AT' (β1): {beta_1:.2f}")

y_pred = model.predict(X)
n = len(y)
SE_beta_1 = np.sqrt(np.sum((y - y_pred)**2) / (n - 2)) / np.sqrt(np.sum((X['AT'] - np.mean(X['AT']))**2))

# Obtenir la valeur critique t pour un intervalle de confiance de 95%
t_critical = stats.t.ppf(1 - 0.025, df=n-2)

# Calculer l'intervalle de confiance pour β1
CI_lower = beta_1 - t_critical * SE_beta_1
CI_upper = beta_1 + t_critical * SE_beta_1

print(f"L'intervalle de confiance à 95% pour β1 est de {CI_lower:.2f} à {CI_upper:.2f}.")

# Continuation du code précédent pour calculer SE(β̂1)
# ...

# Calculer la statistique t pour β1
t_statistic = beta_1 / SE_beta_1

# Calculer la p-value pour le test t bilatéral
p_value = (1 - stats.t.cdf(abs(t_statistic), df=n-2)) * 2

print(f"Statistique t pour β1: {t_statistic:.2f}")
print(f"P-value pour le test d'hypothèse que β1 = 0: {p_value:.4f}")

# Conclure si β1 est significativement non nul
alpha = 0.05  # Seuil de signification de 5%
if p_value < alpha:
    print("Nous rejetons l'hypothèse nulle, ce qui suggère que β1 est significativement non nul.")
else:
    print("Nous ne pouvons pas rejeter l'hypothèse nulle, ce qui suggère que β1 pourrait être nul.")

from sklearn.metrics import r2_score

# Prédire les valeurs de EP en utilisant le modèle
y_pred = model.predict(X)

# Calculer le R²
r_squared = r2_score(y, y_pred)

print(f"Le coefficient de détermination R² est: {r_squared:.2f}")
