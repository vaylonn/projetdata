import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import t

# Charger le jeu de données
df = pd.read_csv('CCPP_data.txt', delimiter='\t')

# Calculer la matrice de corrélation pour l'ensemble du jeu de données
corr_matrix = df.corr()

# Afficher la matrice de corrélation
print(f"La matrice de corrélation pour l'ensemble du jeu de données est :\n{corr_matrix}\n")

# Identifier la variable la plus corrélée à la variable cible EP
most_correlated_var = corr_matrix['EP'].sort_values(ascending=False).index[1]

# Ajuster un modèle de régression linéaire simple en utilisant la variable la plus corrélée
X = df[most_correlated_var].values.reshape(-1, 1)
Y = df['EP'].values.reshape(-1, 1)
reg = LinearRegression().fit(X, Y)

# Afficher les estimations des coefficients
beta_0 = reg.intercept_[0]
beta_1 = reg.coef_[0][0]
print(f"Les estimations des coefficients sont : beta_0 = {beta_0:.2f}, beta_1 = {beta_1:.2f}\n")

# Calculer l'intervalle de confiance à 95 % pour beta_1
alpha = 0.05
n = len(X)
t_alpha_2 = t.ppf(1 - alpha / 2, n - 2)
s = np.sqrt(np.sum((Y - reg.predict(X)) ** 2) / (n - 2))
s_x = np.std(X, ddof=1)
x_bar = np.mean(X)
ci_lower = beta_1 - t_alpha_2 * s / (s_x * np.sqrt(n))
ci_upper = beta_1 + t_alpha_2 * s / (s_x * np.sqrt(n))
print(f"L'intervalle de confiance à 95 % pour beta_1 est : ({ci_lower:.2f}, {ci_upper:.2f})\n")

# Tester l'hypothèse de pente nulle pour beta_1
t_stat = beta_1 / (s / (s_x * np.sqrt(n)))
p_value = 2 * (1 - t.cdf(abs(t_stat), n - 2))
print(f"La valeur-p pour le test d'hypothèse de pente nulle est : {p_value:.4f}")
if p_value < alpha:
    print("Le prédicteur a un impact significatif sur la production d'énergie électrique nette horaire (EP).\n")
else:
    print("Le prédicteur n'a pas d'impact significatif sur la production d'énergie électrique nette horaire (EP).\n")

# Calculer le coefficient de détermination R^2
R2 = reg.score(X, Y)
print(f"Le coefficient de détermination R^2 est : {R2:.4f}")
if R2 > 0.5:
    print("Ce modèle est adapté pour prédire la production d'énergie électrique nette horaire (EP).\n")
else:
    print("Ce modèle n'est pas adapté pour prédire la production d'énergie électrique nette horaire (EP).\n")
