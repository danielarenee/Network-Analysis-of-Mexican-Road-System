import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# 1. Descargar datos históricos de varias acciones
#tickers = ["AAPL", "MSFT","TSLA", "GOOGL", "AMZN"]
#tickers = ["JNJ", "KO", "PG", "PFE", "WMT"]
#tickers = ["GOOGL", "AAPL", "MSFT"]
tickers = ["TSLA", "KO", "PFE"]


data = yf.download(tickers, start="2023-01-01", end="2025-01-01")["Close"]

# 2. Calcular rendimientos diarios
returns = data.pct_change().dropna()

# 3. Calcular estadísticas del portafolio
rendimientos_esperados = returns.mean()
cov_matrix = returns.cov()

# ---  PORTAFOLIO MÍNIMA VARIANZA  ---
def riesgo_portafolio(pesos, cov_matrix):
    return np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))

def restriccion_suma(pesos):
    return np.sum(pesos) - 1  # Los pesos deben sumar 100%

limites = [(0, 0.5) for _ in tickers]  # Pesos entre 0% y 50%
pesos_iniciales = np.array([1/len(tickers)] * len(tickers))

# Optimización de mínima varianza
opt_res_min_var = opt.minimize(
    riesgo_portafolio, pesos_iniciales,
    args=(cov_matrix,),
    method="SLSQP",
    constraints={"type": "eq", "fun": restriccion_suma},
    bounds=limites
)

pesos_min_var = opt_res_min_var.x
rend_min_var = np.dot(pesos_min_var, rendimientos_esperados)
riesgo_min_var = riesgo_portafolio(pesos_min_var, cov_matrix)

# ---  PORTAFOLIO MÁXIMO SHARPE  ---
def sharpe_ratio(pesos, rendimientos_esperados, cov_matrix, rf=0):
    rend_portafolio = np.dot(pesos, rendimientos_esperados)
    riesgo_portafolio = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
    return -(rend_portafolio - rf) / riesgo_portafolio  # Negativo porque scipy minimiza

opt_res_sharpe = opt.minimize(
    sharpe_ratio, pesos_iniciales,
    args=(rendimientos_esperados, cov_matrix),
    method="SLSQP",
    constraints={"type": "eq", "fun": restriccion_suma},
    bounds=limites
)

pesos_max_sharpe = opt_res_sharpe.x
rend_max_sharpe = np.dot(pesos_max_sharpe, rendimientos_esperados)
riesgo_max_sharpe = riesgo_portafolio(pesos_max_sharpe, cov_matrix)


# ---  PORTAFOLIO MÁXIMO RATIO SORTINO  ---

#funcion objetivo: negativo del ratio sortino
def sortino_ratio(pesos, returns, rendimientos_esperados, rf=0):
    rp = np.dot(pesos, rendimientos_esperados) 
    rend_diario_portafolio = np.dot(returns, pesos)#  vector de rendimientos diarios
    rend_negativos = rend_diario_portafolio[rend_diario_portafolio < 0] #filtrar rendimientos negativos

    # desviacion a la baja
    riesgo_downside = np.std(rend_negativos, ddof=1)  # desviación estándar muestral
    return -(rp - rf) / riesgo_downside


opt_res_sortino = opt.minimize(
    sortino_ratio,
    pesos_iniciales,
    args=(returns, rendimientos_esperados),
    method="SLSQP",
    constraints={"type": "eq", "fun": restriccion_suma},
    bounds=limites
)

pesos_sortino = opt_res_sortino.x
rend_sortino = np.dot(pesos_sortino, rendimientos_esperados)
riesgo_sortino = riesgo_portafolio(pesos_sortino, cov_matrix)  



# ---  PORTAFOLIO EQUILIBRADO  ---
pesos_equilibrado = (pesos_min_var + pesos_max_sharpe) / 2
rend_equilibrado = np.dot(pesos_equilibrado, rendimientos_esperados)
riesgo_equilibrado = riesgo_portafolio(pesos_equilibrado, cov_matrix)


# ---  SIMULACIÓN DE PORTAFOLIOS ALEATORIOS ---
num_portafolios = 10000
resultados = np.zeros((3, num_portafolios))

np.random.seed(23)
for i in range(num_portafolios):
    pesos = np.random.random(len(tickers))
    pesos /= np.sum(pesos)  # Normalizar para que sumen 100%
    
    rendimiento_portafolio = np.dot(pesos, rendimientos_esperados)
    riesgo_portafolio_val = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
    
    resultados[0, i] = rendimiento_portafolio
    resultados[1, i] = riesgo_portafolio_val
    resultados[2, i] = rendimiento_portafolio / riesgo_portafolio_val  # Ratio Sharpe
    

# ---  GRAFICAR LA FRONTERA EFICIENTE  ---

plt.figure(figsize=(10, 6))
plt.scatter(resultados[1], 
            resultados[0], 
            c=resultados[2], 
            cmap="coolwarm", 
            alpha=0.6, 
            label="Portafolios Simulados")
plt.scatter(riesgo_min_var, rend_min_var, color="blue", marker="*", s=200, label="Min Varianza")
plt.scatter(riesgo_max_sharpe, rend_max_sharpe, color="red", marker="*", s=200, label="Máx. Sharpe")
plt.scatter(riesgo_sortino, rend_sortino, color="green", marker="*", s=200, label="Máx Ratio Sortino")
plt.scatter(riesgo_equilibrado, rend_equilibrado, color="yellow", marker="*", s=200, label="Portafolio Equilibrado")


plt.xlabel("Riesgo (Desviación Estándar)")
plt.ylabel("Rendimiento Esperado")
plt.title("Frontera Eficiente con Portafolio Personalizado")
plt.legend()
plt.show()


# ---  IMPRIMIR RESULTADOS DE TODOS LOS PORTAFOLIOS  ---

print("\n🔵 Pesos del portafolio de mínima varianza:")
for asset, weight in zip(tickers, pesos_min_var):
    print(f"{asset}: {weight:.2%}")

print(f"\n Rendimiento esperado (diario): {rend_min_var:.2%}")
print(f" Riesgo esperado (diario): {riesgo_min_var:.2%}")

print("\n🔴 Pesos del portafolio de máximo Sharpe:")
for asset, weight in zip(tickers, pesos_max_sharpe):
    print(f"{asset}: {weight:.2%}")

print(f"\n Rendimiento esperado (diario): {rend_max_sharpe:.2%}")
print(f" Riesgo esperado (diario): {riesgo_max_sharpe:.2%}")

print(f"\n🟢 Pesos del portafolio de máximo Ratio Sortino:")
for asset, weight in zip(tickers, pesos_sortino):
    print(f"{asset}: {weight:.2%}")

print(f"\n Rendimiento esperado (diario): {rend_sortino:.2%}")
print(f" Riesgo esperado (diario): {riesgo_sortino:.2%}")

print("\n🟡 Pesos del Portafolio Equilibrado (50% Mín. Var. + 50% Máx. Sharpe):")
for asset, weight in zip(tickers, pesos_equilibrado):
    print(f"{asset}: {weight:.2%}")

print(f"\n Rendimiento esperado (diario): {rend_equilibrado:.2%}")
print(f" Riesgo esperado (diario): {riesgo_equilibrado:.2%}")


# ---  CALCULAR VOLATILIDAD ANUALIZADA  ---

volatilidad_min_var = riesgo_min_var * np.sqrt(252)
volatilidad_max_sharpe = riesgo_max_sharpe * np.sqrt(252)
volatilidad_sortino = riesgo_sortino * np.sqrt(252)
volatilidad_equilibrado = riesgo_equilibrado * np.sqrt(252)

print("\n Volatilidad anualizada de cada portafolio:")
print(f"🔵 Mínima Varianza: {volatilidad_min_var:.2%}")
print(f"🔴 Máximo Sharpe: {volatilidad_max_sharpe:.2%}")
print(f"🟢 Máximo Ratio Sortino: {volatilidad_sortino:.2%}")
print(f"🟡 Equilibrado: {volatilidad_equilibrado :.2%}")


# ---  CONVERTIR RENDIMIENTO ESPERADO DIARIO A ANUAL  ---

rend_anual_min_var = ((1 + rend_min_var) ** 252) - 1
rend_anual_max_sharpe = (1 + rend_max_sharpe) ** 252 - 1
rend_anual_sortino = (1 + rend_sortino) ** 252 - 1
rend_anual_equilibrado = (1 + rend_equilibrado) ** 252 - 1

print("\n Rendimiento esperado anualizado de cada portafolio:")
print(f"🔵 Mínima Varianza: {rend_anual_min_var:.2%}")
print(f"🔴 Máximo Sharpe: {rend_anual_max_sharpe:.2%}")
print(f"🟢 Máximo Ratio Sortino: {rend_anual_sortino:.2%}")
print(f"🟡 Equilibrado: {rend_anual_equilibrado:.2%}")

