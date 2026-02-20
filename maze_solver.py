from __future__ import annotations

import os
import heapq
from abc import ABC, abstractmethod
from collections import deque
from enum import Enum
from typing import List, Tuple, Optional, Set, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

from neural_network_class import NeuralNetwork

plt.rcParams["figure.dpi"] = 120

State = Tuple[int, int]  # (row, col)
Action = Tuple[int, int]  # delta (dr, dc)
FOUR_CONNECTED: List[Action] = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class CellType(Enum):
    """Tipos de celda en la grilla discretizada."""
    FREE  = 0
    WALL  = 1
    START = 2
    GOAL  = 3


class SearchProblem(ABC):
    """Interfaz genérica para definir un problema de búsqueda formal."""

    @abstractmethod
    def initial_state(self) -> State:
        """Retorna el estado inicial."""
        ...

    @abstractmethod
    def goal_test(self, state: State) -> bool:
        """Retorna True si `state` es un estado meta."""
        ...

    @abstractmethod
    def actions(self, state: State) -> List[Action]:
        """Retorna la lista de acciones válidas desde `state`."""
        ...

    @abstractmethod
    def result(self, state: State, action: Action) -> State:
        """Retorna el estado resultante de aplicar `action` en `state`."""
        ...

    @abstractmethod
    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        """Retorna el costo de moverse de `state` a `next_state` vía `action`."""
        ...


class Node:
    """Nodo del árbol de búsqueda.

    Attributes
    ----------
    state : State
        Estado (fila, columna) representado por este nodo.
    parent : Node | None
        Nodo padre (None para la raíz).
    action : Action | None
        Acción que llevó del padre a este nodo.
    path_cost : float
        Costo acumulado g(n) desde el inicio hasta este nodo.
    """

    __slots__ = ("state", "parent", "action", "path_cost")

    def __init__(
        self,
        state: State,
        parent: Optional["Node"] = None,
        action: Optional[Action] = None,
        path_cost: float = 0.0,
    ) -> None:
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    # Para uso con heapq (A*)
    def __lt__(self, other: "Node") -> bool:
        return self.path_cost < other.path_cost

    def path(self) -> List[State]:
        """Reconstruye la ruta desde la raíz hasta este nodo."""
        states: List[State] = []
        node: Optional[Node] = self
        while node is not None:
            states.append(node.state)
            node = node.parent
        states.reverse()
        return states


class SearchResult:
    """Contenedor para los resultados de una búsqueda."""

    def __init__(
        self,
        path: List[State],
        path_cost: float,
        nodes_explored: int,
        success: bool,
    ) -> None:
        self.path = path
        self.path_cost = path_cost
        self.nodes_explored = nodes_explored
        self.success = success

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"SearchResult({status}, pasos={len(self.path)-1}, "
            f"costo={self.path_cost:.1f}, explorados={self.nodes_explored})"
        )


def classify_color(r: float, g: float, b: float) -> CellType:
    """Clasifica un color RGB promedio en un tipo de celda.

    Se usan umbrales relajados para tolerar artefactos de compresión en BMP/PNG.
    """
    # Negro  →  pared (todos los canales bajos)
    if r < 50 and g < 50 and b < 50:
        return CellType.WALL

    # Rojo  →  inicio (R dominante)
    if r > 150 and g < 100 and b < 100:
        return CellType.START

    # Verde  →  meta (G dominante)
    if g > 150 and r < 100 and b < 100:
        return CellType.GOAL

    # Todo lo demás → camino libre (incluyendo blanco)
    return CellType.FREE


def discretize(image: np.ndarray, tile_size: int = 10) -> np.ndarray:
    """Convierte una imagen RGB en una grilla discreta de CellType.

    Parameters
    ----------
    image : np.ndarray
        Arreglo (H, W, 3) con valores uint8.
    tile_size : int
        Número de píxeles por lado de cada tile.

    Returns
    -------
    grid : np.ndarray
        Matriz 2-D de objetos CellType con forma (rows, cols).
    """
    h, w, _ = image.shape
    rows = h // tile_size
    cols = w // tile_size

    grid = np.empty((rows, cols), dtype=object)

    for r in range(rows):
        for c in range(cols):
            # Extraer el tile
            tile = image[
                r * tile_size : (r + 1) * tile_size,
                c * tile_size : (c + 1) * tile_size,
            ]
            # Color promedio del tile
            avg = tile.mean(axis=(0, 1))  # (R, G, B) float
            grid[r, c] = classify_color(avg[0], avg[1], avg[2])

    return grid


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convierte la grilla de CellType a una imagen RGB para visualización."""
    color_map = {
        CellType.FREE:  [255, 255, 255],  # blanco
        CellType.WALL:  [0,   0,   0  ],  # negro
        CellType.START: [255, 0,   0  ],  # rojo
        CellType.GOAL:  [0,   255, 0  ],  # verde
    }
    rows, cols = grid.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            rgb[r, c] = color_map[grid[r, c]]
    return rgb


def extract_key_positions(
    grid: np.ndarray,
) -> Tuple[Tuple[int, int], List[Tuple[int, int]], int]:
    """Extrae las coordenadas de inicio, metas y cuenta de paredes.

    Si el área roja ocupa varios tiles, se calcula el centroide y los
    tiles restantes se reclasifican como FREE.

    Parameters
    ----------
    grid : np.ndarray
        Matriz 2-D de CellType.

    Returns
    -------
    start : (row, col)
    goals : list of (row, col)
    wall_count : int
    """
    start_tiles: List[Tuple[int, int]] = []
    goals: List[Tuple[int, int]] = []
    wall_count = 0

    rows, cols = grid.shape
    for r in range(rows):
        for c in range(cols):
            cell = grid[r, c]
            if cell == CellType.START:
                start_tiles.append((r, c))
            elif cell == CellType.GOAL:
                goals.append((r, c))
            elif cell == CellType.WALL:
                wall_count += 1

    assert len(start_tiles) > 0, "No se encontró un punto de inicio (rojo) en la imagen."
    assert len(goals) > 0, "No se encontró ninguna meta (verde) en la imagen."

    # Consolidar tiles START en un único centroide
    if len(start_tiles) == 1:
        start = start_tiles[0]
    else:
        avg_r = int(round(sum(r for r, _ in start_tiles) / len(start_tiles)))
        avg_c = int(round(sum(c for _, c in start_tiles) / len(start_tiles)))
        start = (avg_r, avg_c)
        print(f"  ⚠ {len(start_tiles)} tiles START detectados → centroide en {start}")

    # Reclasificar los tiles START sobrantes como FREE
    for r, c in start_tiles:
        if (r, c) != start:
            grid[r, c] = CellType.FREE
    grid[start[0], start[1]] = CellType.START

    return start, goals, wall_count


def manhattan_heuristic(
    state: State, goals: Set[Tuple[int, int]]
) -> float:
    """Distancia Manhattan mínima desde `state` a cualquier meta.

    Es admisible (nunca sobreestima) y consistente para movimiento 4-conexo.
    """
    r, c = state
    return min(abs(r - gr) + abs(c - gc) for gr, gc in goals)


def graph_search(
    problem: SearchProblem,
    frontier,               # deque (BFS) o list (DFS)
    is_lifo: bool = False,  # True → DFS, False → BFS
) -> SearchResult:
    """Algoritmo genérico de graph search.

    Parameters
    ----------
    problem : SearchProblem
        Instancia del problema.
    frontier : collection
        Estructura de datos para la frontera (deque o list).
    is_lifo : bool
        Si True, usa pop() (LIFO/DFS). Si False, usa popleft() (FIFO/BFS).

    Returns
    -------
    SearchResult
    """
    start_node = Node(state=problem.initial_state())
    frontier.append(start_node)
    explored: Set[State] = set()
    nodes_explored = 0

    while frontier:
        # Extraer nodo de la frontera
        node: Node = frontier.pop() if is_lifo else frontier.popleft()

        # Si ya fue explorado, saltar
        if node.state in explored:
            continue

        explored.add(node.state)
        nodes_explored += 1

        # Test de meta
        if problem.goal_test(node.state):
            return SearchResult(
                path=node.path(),
                path_cost=node.path_cost,
                nodes_explored=nodes_explored,
                success=True,
            )

        # Expandir
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            if child_state not in explored:
                child_cost = node.path_cost + problem.step_cost(
                    node.state, action, child_state
                )
                child_node = Node(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=child_cost,
                )
                frontier.append(child_node)

    # No se encontró solución
    return SearchResult(path=[], path_cost=float("inf"), nodes_explored=nodes_explored, success=False)


def bfs(problem: SearchProblem) -> SearchResult:
    """Breadth-First Search: frontera FIFO."""
    return graph_search(problem, frontier=deque(), is_lifo=False)


def dfs(problem: SearchProblem) -> SearchResult:
    """Depth-First Search: frontera LIFO."""
    return graph_search(problem, frontier=[], is_lifo=True)


def astar(
    problem: "MazeProblem",
    heuristic,
) -> SearchResult:
    """Algoritmo A* usando cola de prioridad (min-heap).

    Parameters
    ----------
    problem : MazeProblem
    heuristic : callable(state, goals) -> float

    Returns
    -------
    SearchResult
    """
    start_node = Node(state=problem.initial_state(), path_cost=0.0)
    goals = problem._goals  # acceso directo para la heurística

    # (f(n), counter, node)  —  counter rompe empates determinista
    counter = 0
    open_heap: List[Tuple[float, int, Node]] = []
    h0 = heuristic(start_node.state, goals)
    heapq.heappush(open_heap, (h0, counter, start_node))

    # g-values: mejor costo conocido para llegar a cada estado
    best_g: Dict[State, float] = {start_node.state: 0.0}
    nodes_explored = 0

    while open_heap:
        f_val, _, node = heapq.heappop(open_heap)

        # Si ya procesamos este estado con menor costo, saltar
        if node.path_cost > best_g.get(node.state, float("inf")):
            continue

        nodes_explored += 1

        # Test de meta
        if problem.goal_test(node.state):
            return SearchResult(
                path=node.path(),
                path_cost=node.path_cost,
                nodes_explored=nodes_explored,
                success=True,
            )

        # Expandir
        for action in problem.actions(node.state):
            child_state = problem.result(node.state, action)
            new_g = node.path_cost + problem.step_cost(
                node.state, action, child_state
            )

            # Solo explorar si mejoramos el costo previo
            if new_g < best_g.get(child_state, float("inf")):
                best_g[child_state] = new_g
                h = heuristic(child_state, goals)
                f = new_g + h
                counter += 1
                child_node = Node(
                    state=child_state,
                    parent=node,
                    action=action,
                    path_cost=new_g,
                )
                heapq.heappush(open_heap, (f, counter, child_node))

    return SearchResult(
        path=[], path_cost=float("inf"), nodes_explored=nodes_explored, success=False
    )


class MazeProblem(SearchProblem):
    """Problema de búsqueda en un laberinto discretizado.

    Parameters
    ----------
    grid : np.ndarray
        Matriz 2-D de CellType producida por `discretize()`.
    start : (row, col)
        Coordenadas del nodo de inicio.
    goals : list of (row, col)
        Coordenadas de los nodos meta.
    """

    def __init__(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goals: List[Tuple[int, int]],
        image: Optional[np.ndarray] = None,
        nn: Optional[NeuralNetwork] = None,
        cost: Optional[dict] = None,
        tile_size: int = 10,
    ) -> None:
        self.grid = grid
        self.rows, self.cols = grid.shape
        self._start = start
        self._goals: Set[Tuple[int, int]] = set(goals)
        self.image = image
        self.nn = nn
        self.cost = cost
        self.tile_size = tile_size

    def initial_state(self) -> State:
        return self._start

    def goal_test(self, state: State) -> bool:
        return state in self._goals

    def actions(self, state: State) -> List[Action]:
        r, c = state
        valid: List[Action] = []
        for dr, dc in FOUR_CONNECTED:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr, nc] != CellType.WALL:
                    valid.append((dr, dc))
        return valid

    def result(self, state: State, action: Action) -> State:
        return (state[0] + action[0], state[1] + action[1])

    def step_cost(self, state: State, action: Action, next_state: State) -> float:
        """
        Task 2.2: Costo dinámico basado en predicción de red neuronal.
        Si no hay red neuronal, retorna costo uniforme = 1.0
        
        1. Ubica el tile destino en la imagen original
        2. Extrae el color RGB promedio del tile
        3. Predice la clase con la red neuronal
        4. Convierte a costo usando el diccionario
        """
        # Si no hay red neuronal, retornar costo uniforme (Task 1.3)
        if self.nn is None or self.image is None or self.cost is None:
            return 1.0
        
        # 1. Ubicar el tile en píxeles
        row_grid, col_grid = next_state
        row_px = row_grid * self.tile_size
        col_px = col_grid * self.tile_size
        
        # 2. Extraer el tile de la imagen original
        tile = self.image[
            row_px : row_px + self.tile_size,
            col_px : col_px + self.tile_size
        ]
        
        # 3. Calcular color promedio y normalizar a [0, 1]
        rgb_promedio = tile.mean(axis=(0, 1))  # (R, G, B) en [0, 255]
        rgb_normalizado = rgb_promedio / 255.0  # [0, 1]
        
        # 4. Predecir clase con la red neuronal
        clase_idx = self.nn.predict([rgb_normalizado])[0]
        
        # 5. Obtener costo del diccionario (clave = índice de clase)
        costo = self.cost[clase_idx]
        
        return float(costo)
