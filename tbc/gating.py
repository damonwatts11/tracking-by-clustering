"""
tbc/gating.py
=============
Estimación adaptativa (data-driven) de los gating parameters rho_s (espacial)
y rho_m (movimiento), integrada con el pipeline tbc.

Reemplaza los gates fijos tipo "k desviaciones estándar de parámetros de la
simulación" por un estimador robusto calculado de la propia nube observada:

    rho = med(D) + c * MAD(D)

donde D es la muestra de distancias a vecino más cercano (intra-frame para
rho_s, inter-frame para rho_m) medidas con la MÉTRICA DEL TORO, y MAD es la
median absolute deviation (robusta a la contaminación por background noise,
a diferencia de la std).

Además separa dos radios que en instance.py estaban acoplados:

  * rho_gate : radio de admisión de aristas (filtro de RECALL, generoso)
  * rho_cost : cero de la función de costo alpha*(rho_cost - d)

En el pipeline original rho_gate == rho_cost, con la consecuencia de que
TODA arista admitida tiene costo > 0 y GAEC degenera en "componentes
conexas del grafo gateado": basta una cadena de puntos de ruido entre dos
objetos para fusionarlos, y no existe ninguna arista repulsiva que lo
impida. Separando rho_cost < rho_gate, las aristas con
rho_cost < d < rho_gate reciben costo negativo y actúan como evidencia
repulsiva, que es como el multicut/clique-partitioning debe funcionar.

Dependencias: numpy, scipy (cKDTree con boxsize implementa la topología
periódica del toro de forma nativa y en O(n log n)).
"""

import numpy as np
from scipy.spatial import cKDTree

from tbc.instance import (
    nodes_by_frame,
    within_frame_edges,
    between_frame_edges,
    edge_costs,
    TrackingInstance,
)


# ----------------------------------------------------------------------
# 1. Estadísticos robustos
# ----------------------------------------------------------------------

def mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation:  MAD(x) = med(|x_i - med(x)|).

    Análogo robusto de la desviación estándar: la std usa sumas sobre
    todos los datos, así que las distancias enormes que introduce el
    background noise la inflan; la mediana solo mira el dato central
    del ordenamiento, así que el ruido (mientras sea minoría de la
    muestra) no la mueve. Este es el motivo por el que el gate fijo
    "k*std" colapsaba al subir el ruido: el umbral crecía con el ruido.
    """
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def robust_threshold(sample: np.ndarray, c: float) -> float:
    """rho = med + c*MAD.  c es el único hiperparámetro (adimensional)."""
    return float(np.median(sample) + c * mad(sample))


# ----------------------------------------------------------------------
# 2. Muestras de distancias a vecino más cercano (métrica del toro)
# ----------------------------------------------------------------------
# cKDTree(pos, boxsize=L) construye el árbol con condiciones de frontera
# periódicas: query() devuelve directamente distancias mínimas-imagen,
# consistentes con tbc.geometry.torus_distance, sin materializar la
# matriz n x n.  Requiere puntos en [0, L), garantizado por wrap().

def nn_distances_intra(points: np.ndarray, times: np.ndarray,
                       T: int, L: float) -> np.ndarray:
    """
    d_intra(p) = distancia (en el toro) de cada punto a su vecino más
    cercano DENTRO de su propio frame. Muestra agregada sobre los T
    frames. Es la cantidad cuya distribución mezcla la escala interna
    de los objetos (valores bajos) con la escala del ruido (altos).
    """
    out = []
    for node_ids in nodes_by_frame(times, T):
        if len(node_ids) < 2:
            continue
        pos = points[node_ids]
        tree = cKDTree(pos, boxsize=L)
        # k=2: el vecino más cercano de un punto es él mismo (dist 0)
        dists, _ = tree.query(pos, k=2)
        out.append(dists[:, 1])
    return np.concatenate(out) if out else np.array([])


def nn_distances_inter(points: np.ndarray, times: np.ndarray,
                       T: int, L: float) -> np.ndarray:
    """
    d_inter(p) = distancia (en el toro) de cada punto del frame t a su
    vecino más cercano en el frame t+1.  Estima el desplazamiento por
    paso de tiempo.

    NOTA: solo pares FÍSICAMENTE consecutivos (t, t+1) con t <= T-2.
    El par cíclico (T-1, 0) que usa el tracker NO entra en la muestra:
    la simulación corre hacia adelante sin periodicidad temporal, así
    que ese par no representa un desplazamiento de un paso y meterlo
    contaminaría el estimador con distancias arbitrariamente grandes.
    """
    frames = nodes_by_frame(times, T)
    out = []
    for t in range(T - 1):
        ids_t, ids_next = frames[t], frames[t + 1]
        if len(ids_t) == 0 or len(ids_next) == 0:
            continue
        tree = cKDTree(points[ids_next], boxsize=L)
        dists, _ = tree.query(points[ids_t], k=1)
        out.append(dists)
    return np.concatenate(out) if out else np.array([])


# ----------------------------------------------------------------------
# 3. Estimación de los gates
# ----------------------------------------------------------------------

def estimate_gates(points: np.ndarray, times: np.ndarray, T: int, L: float,
                   c_spatial: float = 3.0, c_motion: float = 3.0):
    """
    rho_s = med(D_s) + c_spatial * MAD(D_s)
    rho_m = med(D_m) + c_motion  * MAD(D_m)

    Devuelve (rho_s, rho_m). Equivariantes por escala: si la nube se
    reescala por lambda, ambos gates se reescalan por lambda solos.
    """
    d_s = nn_distances_intra(points, times, T, L)
    d_m = nn_distances_inter(points, times, T, L)
    return robust_threshold(d_s, c_spatial), robust_threshold(d_m, c_motion)


def estimate_gates_from_dataset(ds, c_spatial: float = 3.0,
                                c_motion: float = 3.0):
    """Conveniencia: lee points/times/T/L directamente del SyntheticDataset."""
    return estimate_gates(ds.points, ds.times,
                          ds.config.n_timesteps, ds.config.cube_size,
                          c_spatial=c_spatial, c_motion=c_motion)


# ----------------------------------------------------------------------
# 4. Pre-filtro de ruido (opcional, recomendado con mucho background)
# ----------------------------------------------------------------------

def core_point_mask(points: np.ndarray, times: np.ndarray, T: int, L: float,
                    rho_s: float, min_neighbors: int = 1) -> np.ndarray:
    """
    Máscara booleana (M,): True si el punto tiene >= min_neighbors
    vecinos a distancia <= rho_s en su propio frame.

    Fundamento: un punto de objeto pertenece a un cluster denso (en tu
    simulación, ~n_inliers_per_sphere companeros por frame), así que
    tiene vecinos dentro del gate espacial. Un punto sin vecinos solo
    puede aportar aristas espurias o quedar singleton: excluirlo del
    grafo elimina los "puentes de ruido" entre objetos ANTES de que el
    greedy pueda cometer el error (misma lógica core/noise de DBSCAN).

    Los puntos filtrados NO se eliminan del dataset: solo se excluyen
    de la construcción de aristas y quedan como singletons, de modo que
    el array de labels sigue alineado con ds.labels para el VI.
    """
    mask = np.zeros(len(points), dtype=bool)
    for node_ids in nodes_by_frame(times, T):
        if len(node_ids) < 2:
            continue
        pos = points[node_ids]
        tree = cKDTree(pos, boxsize=L)
        # cuenta vecinos dentro de rho_s (excluyéndose a sí mismo)
        counts = tree.query_ball_point(pos, r=rho_s, return_length=True) - 1
        mask[node_ids] = counts >= min_neighbors
    return mask


# ----------------------------------------------------------------------
# 5. Ensamblado: build_instance con gates adaptativos y costos con signo
# ----------------------------------------------------------------------

def build_instance_adaptive(ds,
                            c_gate_spatial: float = 3.0,
                            c_gate_motion: float = 3.0,
                            c_cost_spatial: float = 0.5,
                            c_cost_motion: float = 0.5,
                            alpha_in: float = 1.0,
                            alpha_mot: float = 1.0,
                            min_neighbors: int | None = 3,
                            cyclic: bool = True,
                            verbose: bool = True) -> TrackingInstance:
    """
    Igual que tbc.instance.build_instance, pero:

      1) rho_gate y rho_cost se ESTIMAN de la data (med + c*MAD) en vez
         de fijarse a mano;
      2) rho_gate (c_gate_*) y rho_cost (c_cost_*) están SEPARADOS:
           - el gate admite aristas hasta med + c_gate*MAD  (recall)
           - el costo cruza cero en    med + c_cost*MAD     (precisión)
         con c_cost < c_gate, de modo que existen aristas de costo
         negativo y GAEC resuelve un multicut de verdad en lugar de
         devolver componentes conexas;
      3) opcionalmente excluye puntos de ruido del grafo (min_neighbors;
         None para desactivar).

    Requisito: c_cost_* <= c_gate_* (si no, el gate recorta las aristas
    negativas y volvemos al problema original).
    """
    assert c_cost_spatial <= c_gate_spatial and c_cost_motion <= c_gate_motion, \
        "c_cost debe ser <= c_gate: el gate debe admitir aristas de costo negativo"

    T = ds.config.n_timesteps
    L = ds.config.cube_size
    points, times = ds.points, ds.times

    # ---- 1) muestras d1 y los cuatro radios --------------------------
    d_s = nn_distances_intra(points, times, T, L)
    d_m = nn_distances_inter(points, times, T, L)

    rho_gate_in  = robust_threshold(d_s, c_gate_spatial)
    rho_gate_mot = robust_threshold(d_m, c_gate_motion)
    rho_cost_in  = robust_threshold(d_s, c_cost_spatial)
    rho_cost_mot = robust_threshold(d_m, c_cost_motion)

    # ---- 2) agrupar nodos por frame; pre-filtro de ruido opcional ----
    frames = nodes_by_frame(times, T)
    if min_neighbors is not None:
        mask = core_point_mask(points, times, T, L,
                               rho_gate_in, min_neighbors=min_neighbors)
        frames = [ids[mask[ids]] for ids in frames]
        n_filtered = int((~mask).sum())
    else:
        n_filtered = 0

    # ---- 3) aristas con el gate generoso, costos con el cero interno -
    w_edges, w_dists = within_frame_edges(points, frames, rho_gate_in, L)
    w_costs = edge_costs(w_dists, rho_cost_in, alpha_in)

    b_edges, b_dists = between_frame_edges(points, frames, rho_gate_mot,
                                           L, T, cyclic=cyclic)
    b_costs = edge_costs(b_dists, rho_cost_mot, alpha_mot)

    edges = np.concatenate([w_edges, b_edges], axis=0)
    costs = np.concatenate([w_costs, b_costs], axis=0)
    kind = np.concatenate([np.zeros(len(w_edges), dtype=int),
                           np.ones(len(b_edges), dtype=int)])

    instance = TrackingInstance(points=points, times=times, edges=edges,
                                costs=costs, kind=kind,
                                n_nodes=len(points), T=T)

    if verbose:
        fp_w = (w_costs > 0).mean() if len(w_costs) else 0.0
        fp_b = (b_costs > 0).mean() if len(b_costs) else 0.0
        print("===== build_instance_adaptive() summary =====")
        print(f"rho_gate_in  = {rho_gate_in:.4f}   rho_cost_in  = {rho_cost_in:.4f}")
        print(f"rho_gate_mot = {rho_gate_mot:.4f}   rho_cost_mot = {rho_cost_mot:.4f}")
        print(f"puntos excluidos por filtro de ruido : {n_filtered}")
        print(f"|E_t| = {len(w_edges)}   |E_t,t+1| = {len(b_edges)}   |E| = {len(edges)}")
        print(f"Fraccion positiva (within)  : {fp_w:.2%}   <- ya NO debe ser 100%")
        print(f"Fraccion positiva (between) : {fp_b:.2%}")
        print("=============================================")

    return instance
