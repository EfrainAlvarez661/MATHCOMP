import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Datos
x1 = np.array([1, 1, 2, 3, 1, 2, 3, 3]).reshape(-1, 1)
x2 = np.array([-1, 0, 0, 1, 1, 2, 2, 0]).reshape(-1, 1)
y = np.array([1.6, 3, 1.1, 1.2, 3.2, 3.3, 1.7, 0.1])

# Combinar x1 y x2 en un solo array de características
X = np.hstack((x1, x2))

# Ajuste lineal
model = LinearRegression()
model.fit(X, y)

# Coeficientes
slope1, slope2 = model.coef_
intercept = model.intercept_

# Calcular el coeficiente de correlación
r_value, _ = pearsonr(y, model.predict(X))

# Resultados
print(f"Ecuación de la recta: y = {slope1:.4f} * x1 + {slope2:.4f} * x2 + {intercept:.4f}")
print(f"Coeficiente de correlación (r): {r_value:.4f}")

# Graficar los datos
plt.scatter(x1, y, color='blue', label='Datos (x1)')
plt.scatter(x2, y, color='red', label='Datos (x2)')
plt.xlabel('Variables')
plt.ylabel('y')
plt.title('Ajuste lineal de datos')
plt.legend()
plt.show()
