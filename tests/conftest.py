"""Gemeinsame Fixtures und Hilfen (DEMO-PLAYBOOK §1b): Mini-Instanzen, an denen jede Einheit von Hand geprüft wird."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def edges_of(pairs):
    """Kantenliste [(a, b), ...] als Arrays (u, v) mit u < v."""
    u = np.array([min(a, b) for a, b in pairs], dtype=int)
    v = np.array([max(a, b) for a, b in pairs], dtype=int)
    return u, v


@pytest.fixture
def one_edge():
    """Zwei Knoten, eine Kante: BP-Handrechnung mit Gruppenanteilen 0,7 / 0,3 und c_aa = 3, c_ab = 1."""
    return 2, *edges_of([(0, 1)])


@pytest.fixture
def two_triangles():
    """Zwei getrennte Dreiecke {0,1,2} und {3,4,5}: die Zerlegung in die beiden Dreiecke hat Modularität 0,5."""
    return 6, *edges_of([(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)])
