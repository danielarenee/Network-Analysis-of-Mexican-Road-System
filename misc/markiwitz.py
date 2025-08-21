#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 18:35:09 2025

@author: danielarenee
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Lista de acciones
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]

# 2. Descargar datos históricos
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")["Close"]

# 3. Calcular los rendimientos diarios
returns = data.pct_change().dropna()

# 4. Mostrar las primeras filas de los rendimientos
print("Rendimientos diarios:")
print(returns.head())

# 5. Calcular la matriz de covarianza
cov_matrix = returns.cov()

# 6. Mostrar la matriz de covarianza
print("\nMatriz de Covarianza:")
print(cov_matrix)

# 7. Crear un mapa de calor con la matriz de covarianza
plt.figure(figsize=(8,6))
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".4f")

plt.title("Matriz de Covarianza entre Acciones")
plt.show()

# 1. Descargar datos históricos de varias acciones
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Lista de acciones
tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]

# Descargar datos de precios de cierre
data = yf.download(tickers, start="2023-01-01", end="2024-01-01")["Close"]

# Calcular los rendimientos diarios
returns = data.pct_change().dropna()  # <<< ESTA LÍNEA DEFINE 'returns'

# Verificar que 'returns' tiene datos antes de continuar
print(returns.head())  # <<< Esto imprimirá los primeros valores para asegurarnos de que está bien definido
