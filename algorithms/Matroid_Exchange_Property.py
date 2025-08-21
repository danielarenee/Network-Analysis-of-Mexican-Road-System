
import networkx as nx
import matplotlib.pyplot as plt
import random
from collections import deque



# crear grafo G - - - - - - - - 

n = 10
p = 0.4
# Grafo de Erdős–Rényi: cada par de nodos se conecta con probabilidad p
G = nx.erdos_renyi_graph(n, p)  

# construir árboles de expansión A y B - - - - - - - - 

A = nx.random_spanning_tree(G, weight=None)  # A
B = nx.random_spanning_tree(G, weight=None)  # B

# calcular A\B y B\A

#
A_edges   = set(A.edges())
B_edges   = set(B.edges())
A_minus_B = A_edges - B_edges # A\B 
B_minus_A = B_edges - A_edges # B\A



# algoritmo de intercambio - - - - - - - - 

def find_all_b(A, B, a):
    
    # 1) quitar a de A
    A_sub = A.copy()
    A_sub.remove_edge(*a)
    u, v = a

    # 2) BFS para etiquetar las dos componentes en A-{a}
    label = {}
    queue = deque([u])
    label[u] = 1
    while queue:
        curr = queue.popleft()
        for w in A_sub.neighbors(curr):
            if w not in label:
                label[w] = 1
                queue.append(w)
    for node in A_sub.nodes():
        if node not in label:
            label[node] = 2

    # 3) Buscar todas las aristas en B\A que crucen componentes distintas
    candidates = []
    for x, y in B.edges():
        if (x, y) not in A.edges() and (y, x) not in A.edges():
            if label.get(x) != label.get(y):
                candidates.append((x, y))

    # mostrar opciones al usuario
    if not candidates:
        print("No se encontraron aristas válidas")
        return None

    print("\nOpciones de aristas b ∈ B\\A que cruzan componentes:")
    for i, edge in enumerate(candidates):
        print(f"{i}: {edge}")

    # Elegir una opción 

    while True:
        idx = int(input("Elige el número de la arista que deseas usar como b: "))
        if 0 <= idx < len(candidates):
            return candidates[idx]
        else:
            print("índice fuera de rango")

# implementacion- - - - - - - - 

a = next(iter(A_minus_B)) #tomar cualquier a de A-B
b = find_all_b(A, B, a)
print(f"Arista a escogida  de  A\B: {a}")
print(f"Arista b escogida ∈ B\A: {b}")
print("Si haces A' = (A -{a})U{b}, obtienes otro spanning tree\n")

# visualizacion - - - - - - - - 

pos = nx.spring_layout(G) 
plt.figure(figsize=(15, 10))

# Grafo completo G
plt.subplot(2, 3, 1)
nx.draw(G, pos,
        with_labels=True,
        node_color='lightgray',
        edge_color='gray',
        node_size=300)
plt.title("G: Grafo completo")

# Árbol A 
plt.subplot(2, 3, 2)
nx.draw(G, pos,
        with_labels=True,
        node_color='white',
        edge_color='lightgray',
        node_size=300)
nx.draw_networkx_edges(A, pos,
                       edge_color='red',
                       width=2)
plt.title("A")

# Árbol B 
plt.subplot(2, 3, 3)
nx.draw(G, pos,
        with_labels=True,
        node_color='white',
        edge_color='lightgray',
        node_size=300)
nx.draw_networkx_edges(B, pos,
                       edge_color='blue',
                       width=2)
plt.title("B")

# A \ B
plt.subplot(2, 3, 4)
subAminusB = nx.Graph()
subAminusB.add_nodes_from(G.nodes())
subAminusB.add_edges_from(A_minus_B)
nx.draw(subAminusB, pos,
        with_labels=True,
        node_color='white',
        edge_color='red',
        node_size=300)
plt.title("A \\ B")

# B \ A
plt.subplot(2, 3, 5)
subBminusA = nx.Graph()
subBminusA.add_nodes_from(G.nodes())
subBminusA.add_edges_from(B_minus_A)
nx.draw(subBminusA, pos,
        with_labels=True,
        node_color='white',
        edge_color='blue',
        node_size=300)
plt.title("B \\ A")

# Destacar a y b en A \ B y B \ A
plt.subplot(2, 3, 6)
nx.draw(G, pos,
        with_labels=True,
        node_color='white',
        edge_color='lightgray',
        node_size=300)
# dibuja 'a' en A\B en rojo grueso
nx.draw_networkx_edges(nx.Graph([a]), pos,
                       edgelist=[a],
                       edge_color='red',
                       width=4, label='a ∈ A\\B')
# dibuja 'b' en B\A en azul grueso
nx.draw_networkx_edges(nx.Graph([b]), pos,
                       edgelist=[b],
                       edge_color='blue',
                       width=4, label='b ∈ B\\A')
plt.title("Intercambio: a (rojo) -> b (azul)")

plt.tight_layout()
plt.show()

#dar la opcion al usuario 
