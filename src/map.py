import numpy as np

def map(map_size: int) -> np.ndarray:
    """
    create a map_size * map_size reward map.
    """
    assert map_size == 4 or map_size == 10, "map_size must be 4 or 10"

    if map_size == 4:
        # 4 holes for 16 grids
        map_array = np.array([[0, 0, 0, 0],
                              [0, -1, 0, -1],
                              [0, 0, 0, -1],
                              [-1, 0, 0, 1]])
    else:
        # 25 holes for 100 grids
        map_array = np.array([[0, 0, -1, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, -1, 0, 0, -1, 0, 0, -1],
                              [-1, 0, 0, 0, -1, 0, 0, -1, 0, 0],
                              [-1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                              [0, -1, 0, 0, 0, -1, 0, 0, 0, 0],
                              [0, 0, -1, 0, 0, 0, -1, 0, 0, 0],
                              [0, -1, 0, -1, 0, -1, 0, 0, 0, 0],
                              [0, 0, 0, -1, 0, 0, -1, -1, 0, 0],
                              [0, 0, -1, 0, -1, 0, -1, 0, -1, 0],
                              [-1, 0, 0, 0, -1, 0, 0, 0, 0, 1]])
    return map_array
