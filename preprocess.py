"""
Convert mesh to graph representation based on computational nodes and mesh elements
"""

import os
import pathlib
from itertools import combinations

import numpy as np
import pandas
from scipy.sparse import coo_matrix, lil_matrix, load_npz
from scipy.sparse import save_npz
from scipy.spatial import cKDTree


def create_As(duplicated_index, size):
    """
    Generate adjacency matrix As from node list.
    :param duplicated_index:Node list, e.g., [[i1,i2,i3],...], where i1,i2,i3 are the same nodes appearing in different mesh elements
    :param size:Matrix size of As
    :return: Generated adjacency matrix As, stored in scipy.sparse.csr_matrix format
    """
    row = []
    col = []
    for item in duplicated_index:
        for j in combinations(item, 2):
            row.append(j[1])
            row.append(j[0])
            col.append(j[0])
            col.append(j[1])

    # Add node self-indices
    row.extend(np.arange(size).tolist())
    col.extend(np.arange(size).tolist())

    num_data = len(row)
    data = np.ones(num_data)
    result = coo_matrix((data, (row, col)), shape=(size, size), dtype=np.int8).tocsr()
    return result

def compute_proximity_adjacency(X, radius, kmax=None):
    coords = np.asarray(X, dtype=float)
    if coords.shape[0] == 0 or radius is None or radius <= 0:
        return coo_matrix((coords.shape[0], coords.shape[0]), dtype=np.int8).tocsr()

    if coords.shape[1] >= 3 and np.allclose(coords[:, 2], coords[0, 2]):
        pts = coords[:, :2]
    else:
        pts = coords[:, :min(3, coords.shape[1])]

    n = pts.shape[0]
    tree = cKDTree(pts)


    if kmax is None:
        pairs = tree.query_pairs(r=radius, output_type='ndarray')  # 形如 [[i,j],...]
        if pairs.size == 0:
            return coo_matrix((n, n), dtype=np.int8).tocsr()
        row = np.concatenate([pairs[:, 0], pairs[:, 1]])
        col = np.concatenate([pairs[:, 1], pairs[:, 0]])
        data = np.ones(len(row), dtype=np.int8)
        return coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int8).tocsr()

    rows_half, cols_half = [], []
    for i in range(n):
        neigh = tree.query_ball_point(pts[i], radius)
        if i in neigh:
            neigh.remove(i)
        if not neigh:
            continue
        if kmax and len(neigh) > kmax:
            d = np.linalg.norm(pts[neigh] - pts[i], axis=1)
            keep_idx = np.argpartition(d, kmax - 1)[:kmax]
            neigh = [neigh[j] for j in keep_idx]
        for j in neigh:
            if j > i: 
                rows_half.append(i)
                cols_half.append(j)

    if not rows_half:
        return coo_matrix((n, n), dtype=np.int8).tocsr()

    row = np.array(rows_half + cols_half, dtype=np.int32)
    col = np.array(cols_half + rows_half, dtype=np.int32)
    data = np.ones(len(row), dtype=np.int8)
    return coo_matrix((data, (row, col)), shape=(n, n), dtype=np.int8).tocsr()


def replace_vertex_matrix(duplicated, shape):
    """
    Replace duplicate nodes with the first node
    :param duplicated: List of duplicate nodes
    :param shape: Matrix shape
    :return: Return sparse matrix
    """
    # Use lil for sparsity changes
    R = lil_matrix(shape, dtype=np.int8)
    remain = []
    for row_col in duplicated:
        R[row_col[0], row_col[1:]] = 1
        remain.extend(row_col[1:])

    size = shape[0]
    R[np.arange(size, dtype=np.int32), np.arange(size, dtype=np.int32)] = 1
    R[remain, remain] = 0
    return R.tocsr()


def _grd2vertex_graph(surfaces, coordinates, no_duplicated=True,
                      distance_threshold=None, proximity_kmax=None):

    """
    Convert 2D Grid mesh data to node-based graph representation, return node feature matrix X and adjacency matrix A
    :param surfaces: Mesh dimensions on each mesh surface
    :param coordinates: Coordinates of mesh points on the mesh
    :param no_duplicated:Whether to keep duplicate nodes
    :return: Matrix X (numpy.ndarray) without duplicate nodes, A (scipy.sparse.csr.csr_matrix),
            and original matrices Xp (numpy.ndarray) and Ad (scipy.sparse.csr.csr_matrix) containing duplicate nodes
    """

    # Number of nodes on each mesh surface
    node_surface = surfaces.prod(axis=1)
    # Total number of mesh points on the entire mesh
    num_nodes = node_surface.sum()

    # Feature matrix Xp containing duplicate nodes
    Xp = pandas.DataFrame(np.zeros([num_nodes, 3]), index=range(num_nodes), columns=['x', 'y', 'z'])
    # Node index counter in mesh
    node_index = 0
    # Used to calculate node indices in X
    last_surface_index = 0
    # Used to index coordinate rows
    x_connect = []
    y_connect = []
    # Construct connection relationships between nodes on each mesh surface
    for i in range(len(node_surface)):
        # Generate x,y,z coordinates for each node
        node_num = node_surface[i]  # Number of nodes on this mesh surface
        coordinates_surface = coordinates.iloc[last_surface_index:last_surface_index + node_num * 3 // 4 + 1, :]
        # Coordinates of all nodes on this mesh surface, first x coordinates, then y coordinates, finally z coordinates
        # Arranged in row order
        last_surface_index += node_num * 3 // 4 + 1  # Calculate starting node index for next mesh surface
        # Expand coordinates into 1D array
        coordinates_surface = coordinates_surface.values.reshape(coordinates_surface.values.size)
        for j in range(3):
            # Assign coordinates in batch to Xp
            Xp.iloc[node_index:node_index + node_num, j] = coordinates_surface[j * node_num:(j + 1) * node_num]

        # Build adjacency matrix A
        # x_dim, y_dim are the dimension sizes of current mesh surface
        x_dim = surfaces.loc[i, 'x_num']
        y_dim = surfaces.loc[i, 'y_num']

        for j in range(node_index, node_index + x_dim * y_dim):
            # Record connection indices of all nodes within current mesh surface
            if (j + 1 - node_index) % x_dim:
                # Connection relationships in x direction
                x_connect.append(j)
                y_connect.append(j + 1)

            if j + x_dim < node_index + x_dim * y_dim:
                # Connection relationships in y direction
                x_connect.append(j)
                y_connect.append(j + x_dim)
        node_index += node_num  # Iterate to next mesh surface

    Ap = coo_matrix((np.ones(len(x_connect), dtype=np.int8), (x_connect, y_connect)), shape=(num_nodes, num_nodes),
                    dtype=np.int8)
    Ap = Ap.tocsr()
    Ap += Ap.transpose()

    if not no_duplicated:
        return Xp, Ap.tocoo()

    # Remove duplicate nodes appearing in different mesh surfaces and build connection relationships between mesh surfaces
    duplicated = Xp.groupby(Xp.columns.to_list()).apply(
        lambda x: list(x.index) if len(x.index) > 1 else None).to_list()
    duplicated_index = [x for x in duplicated if x]
    duplicated_index.sort()  # duplicated_index:List of duplicate nodes

    # Calculate connection relationships between nodes across mesh surfaces, refer to documentation
    As = create_As(duplicated_index, num_nodes)
    An = Ap.dot(As)
    An[An > 1] = 1
    Ad = As.dot(An)
    Ad[Ad > 1] = 1
    # Remove duplicate nodes
    drop_index = [j for i in duplicated_index for j in i[1:]]
    remain_index = sorted(list(set(range(Ad.shape[0])) - set(drop_index)))
    A = Ad[remain_index, :][:, remain_index]

    X = Xp.drop(index=drop_index)

    if distance_threshold is not None and distance_threshold > 0:
        A_prox = compute_proximity_adjacency(
            X.values if isinstance(X, pandas.DataFrame) else X,
            radius=distance_threshold,
            kmax=proximity_kmax
        )
        A = A + A_prox
        A.data[:] = 1
    return X.values, A, [Xp.values, Ad]


def grd2vertex_graph(filename, output_path, store=True,
                     distance_threshold=None, proximity_kmax=None):
    """
    Process grd 2D structured mesh files to generate graph representation based on computational nodes; those without duplicate nodes removed are in Raw folder, those with duplicate nodes removed are in V folder
    :param store:
    :param filename: filename
    :param output_path: output path
    :return: List of feature matrix X (numpy.ndarray) and adjacency matrix A (scipy.sparse.csr.csr_matrix) based on node representation
    """

    # Create new folder if it does not exist
    output_path = pathlib.Path(output_path)
    v_output_path = output_path / 'vertex_graph'
    raw_output_path = output_path / 'raw_vertex_graph'
    if store:
        if not output_path.exists():
            output_path.mkdir()
        if not v_output_path.exists():
            v_output_path.mkdir()
        if not raw_output_path.exists():
            raw_output_path.mkdir()

    FILENAME = os.path.basename(filename)[:-4]
    v_path = v_output_path / (FILENAME + '.npy')
    va_path = v_output_path / (FILENAME + '.npz')
    raw_v_path = raw_output_path / (FILENAME + '.npy')
    raw_va_path = raw_output_path / (FILENAME + '.npz')

    if v_path.exists() and va_path.exists():
        # Return directly if data files exist
        return np.load(v_path), load_npz(va_path)

    with open(filename, 'r') as f:
        # Number of mesh surfaces
        surface_num = int(f.readline().strip())
        # Mesh surface dimensions
        surfaces = pandas.read_csv(filename, skiprows=[0], header=None, nrows=surface_num,
                                   delim_whitespace=True, names=['x_num', 'y_num', 'z_num'])
        # Mesh surface coordinates
        coordinates = pandas.read_csv(filename, skiprows=range(surface_num + 2), header=None, delim_whitespace=True)
        X, A, Raw = _grd2vertex_graph(
            surfaces, coordinates, True,
            distance_threshold=distance_threshold,
            proximity_kmax=proximity_kmax
        )


        if store:
            np.save(v_path, X)
            save_npz(va_path, A)
            np.save(raw_v_path, Raw[0])
            save_npz(raw_va_path, Raw[1])
        return [X, A, Raw[0]]


def _grd2element_graph(X_raw, A, surfaces):
    """
    Implementation of element-based graph data representation
    :param X_raw: Mesh point feature matrix containing duplicate nodes
    :param A: Adjacency matrix based on mesh point representation without duplicate points
    :param surfaces: Dimensions of each mesh surface
    :return: 0-1 normalized element-based feature matrix X and element adjacency matrix A
    """
    surfaces.eval('x_num=x_num-1', inplace=True)
    surfaces.eval('y_num=y_num-1', inplace=True)
    surfaces.eval('quadrangles=x_num*y_num', inplace=True)
    # Total number of mesh elements
    quadrangles = surfaces['quadrangles'].sum()
    # Number of mesh elements on each mesh surface
    quadrangles_perSurface = surfaces['quadrangles']
    # Each row is the index of mesh elements
    X = lil_matrix((quadrangles, 4), dtype=np.int32)
    quadrangle_num = 0
    point_cnt = 0
    for i in range(len(quadrangles_perSurface)):
        x_dim = surfaces.loc[i, 'x_num']
        y_dim = surfaces.loc[i, 'y_num']
        q = quadrangles_perSurface[i]
        points_of_surface = np.array([[[point_cnt + i * (x_dim + 1) + j, point_cnt + i * (x_dim + 1) + j + 1,
                                        point_cnt + i * (x_dim + 1) + j + (x_dim + 1),
                                        point_cnt + i * (x_dim + 1) + j + (x_dim + 1) + 1] for j in
                                       range(x_dim)] for i in range(y_dim)], dtype=np.int32).reshape((q, 4))
        X[quadrangle_num:quadrangle_num + q] = points_of_surface

        quadrangle_num += q
        point_cnt += (x_dim + 1) * (y_dim + 1)

    X = X.toarray()
    node_num = X_raw.shape[0]
    S = lil_matrix((node_num, quadrangles), dtype=np.int8)
    # Cluster assignment matrix S
    S[X, np.arange(quadrangles).reshape(quadrangles, 1)] = 1
    S = S.tocsr()

    X_raw = pandas.DataFrame(X_raw, columns=['x', 'y', 'z'])
    duplicated = X_raw.groupby(X_raw.columns.to_list()).apply(
        lambda x: list(x.index) if len(x.index) > 1 else None).to_list()
    duplicated_index = [x for x in duplicated if x]
    duplicated_index.sort()
    # Node numbering must correspond to each other here

    # Replace all duplicate node indices with the same index here
    R = replace_vertex_matrix(duplicated_index, (node_num, node_num))
    S = R.dot(S)
    S = S[sorted(list(set(range(S.shape[0])) - set([j for i in duplicated_index for j in i[1:]]))), :]
    A = A.tocsr()
    Ae = S.transpose().dot(A).dot(S)
    Ae_data_set = set(Ae.data)
    # Two mesh element edges connected, connection strength is 6
    Ae_data_set -= {6}
    for i in Ae_data_set:
        Ae[Ae == i] = 0
    Ae[Ae == 6] = 1

    # Generate features for each mesh element
    X = pandas.DataFrame(X, columns=['p1', 'p2', 'p3', 'p4'])
    X['v1_x'] = X_raw.iloc[X['p3'], 0].values - X_raw.iloc[X['p1'], 0].values
    X['v1_y'] = X_raw.iloc[X['p3'], 1].values - X_raw.iloc[X['p1'], 1].values
    X['v4_x'] = -X_raw.iloc[X['p2'], 0].values + X_raw.iloc[X['p1'], 0].values
    X['v4_y'] = -X_raw.iloc[X['p2'], 1].values + X_raw.iloc[X['p1'], 1].values
    X['v3_x'] = X_raw.iloc[X['p2'], 0].values - X_raw.iloc[X['p4'], 0].values
    X['v3_y'] = X_raw.iloc[X['p2'], 1].values - X_raw.iloc[X['p4'], 1].values
    X['v2_x'] = -X_raw.iloc[X['p3'], 0].values + X_raw.iloc[X['p4'], 0].values
    X['v2_y'] = -X_raw.iloc[X['p3'], 1].values + X_raw.iloc[X['p4'], 1].values

    # Calculate distortion angle
    X['m1_x'] = 0.5 * (X_raw.iloc[X['p3'], 0].values + X_raw.iloc[X['p1'], 0].values)
    X['m1_y'] = 0.5 * (X_raw.iloc[X['p3'], 1].values + X_raw.iloc[X['p1'], 1].values)
    X['m2_x'] = 0.5 * (X_raw.iloc[X['p2'], 0].values + X_raw.iloc[X['p1'], 0].values)
    X['m2_y'] = 0.5 * (X_raw.iloc[X['p2'], 1].values + X_raw.iloc[X['p1'], 1].values)
    X['m3_x'] = 0.5 * (X_raw.iloc[X['p4'], 0].values + X_raw.iloc[X['p2'], 0].values)
    X['m3_y'] = 0.5 * (X_raw.iloc[X['p4'], 1].values + X_raw.iloc[X['p2'], 1].values)
    X['m4_x'] = 0.5 * (X_raw.iloc[X['p3'], 0].values + X_raw.iloc[X['p4'], 0].values)
    X['m4_y'] = 0.5 * (X_raw.iloc[X['p3'], 1].values + X_raw.iloc[X['p4'], 1].values)

    X['M1_x'] = X['m1_x'] - X['m3_x']
    X['M1_y'] = X['m1_y'] - X['m3_y']
    X['M2_x'] = X['m2_x'] - X['m4_x']
    X['M2_y'] = X['m2_y'] - X['m4_y']
    X['cos_theta'] = (X['M1_x'] * X['M2_x'] + X['M1_y'] * X['M2_y']) / np.sqrt(
        np.power(X['M1_x'], 2) + np.power(X['M1_y'], 2)) / np.sqrt(np.power(X['M2_x'], 2) + np.power(X['M2_y'], 2))

    X['cos_theta'] = np.abs(X['cos_theta'])
    X['theta'] = np.pi / 2 - np.arccos(X['cos_theta'])

    # Element length and edge length ratio
    X['v1'] = np.sqrt(np.power(X['v1_x'], 2) + np.power(X['v1_y'], 2))
    X['v2'] = np.sqrt(np.power(X['v2_x'], 2) + np.power(X['v2_y'], 2))
    X['v3'] = np.sqrt(np.power(X['v3_x'], 2) + np.power(X['v3_y'], 2))
    X['v4'] = np.sqrt(np.power(X['v4_x'], 2) + np.power(X['v4_y'], 2))

    # Element area
    X['area'] = np.multiply(np.multiply(np.sin(np.pi / 2 - X['theta']), X['v1']), X['v2'])

    # Remove irrelevant features
    drop_list = ['p1', 'p2', 'p3', 'p4', 'm1_x', 'm1_y', 'm2_x', 'm2_y', 'm3_x', 'm3_y', 'm4_x', 'm4_y',
                 'M1_x', 'M1_y', 'cos_theta', 'M2_x', 'M2_y', 'delta_theta1', 'delta_theta2',
                 *['cos_theta' + str(i) for i in range(1, 5)],
                 *['theta' + str(i) for i in range(1, 5)], *[f'v{i}_{j}' for i in range(1, 5) for j in ['x', 'y']]]
    X = X.drop(columns=drop_list, errors="ignore")
    X = (X - X.min()) / (X.max() - X.min())

    return X.values, Ae


def grd2element_graph(filename, output_path, store=True):
    filename = pathlib.Path(filename)
    FILENAME = filename.stem

    # Create data file storage folder
    output_path = pathlib.Path(output_path)
    e_output_path = output_path / 'element_graph'
    a_output_path = output_path / 'element_graph'

    if store:
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        if not e_output_path.exists():
            e_output_path.mkdir()

    # Storage path for element feature matrix X
    e_path = e_output_path / (FILENAME + '.npy')
    # Connection matrix A between mesh elements
    a_path = a_output_path / (FILENAME + '.npz')
    if e_path.exists() and a_path.exists():
        # Return directly if data files exist
        return np.load(e_path), load_npz(a_path)

    with open(filename, 'r') as f:
        surface_num = int(f.readline().strip())
        surfaces = pandas.read_csv(filename, skiprows=[0], header=None, nrows=surface_num,
                                   delim_whitespace=True, names=['x_num', 'y_num', 'z_num'])

        raw_v_path = output_path / 'raw_vertex_graph' / (FILENAME + '.npy')
        va_path = output_path / 'vertex_graph' / (FILENAME + '.npz')
        if not raw_v_path.exists() or not va_path.exists():
            # Need to calculate node-based graph data representation here
            _, A, X_raw = grd2vertex_graph(filename, output_path, store=store)
        else:
            X_raw = np.load(raw_v_path)
            A = load_npz(va_path)

        # Generate element-based graph data representation
        S, A = _grd2element_graph(X_raw, A, surfaces)

        # Save data
        if store:
            np.save(e_path, S)
            save_npz(a_path, A, compressed=True)
        return [S, A]


def preprocess_grd_meshes(input_path, output_path, mode=None):
    """
    Generate graph-based grd 2D structured mesh representation
    :param input_path: Mesh file path
    :param output_path: Output file storage path
    :param mode: E, element-based graph representation; V, node-based graph representation
    :return:
    """
    output_path = pathlib.Path(output_path)
    input_path = pathlib.Path(input_path)
    for file in input_path.iterdir():
        if '.grd' == file.suffix:
            print(f'Processing GRD file {file}.')
            if mode == 'E':
                grd2element_graph(file, output_path)
            elif mode == 'V':
                grd2vertex_graph(file, output_path)
            elif mode is None:
                grd2element_graph(file, output_path)
            else:
                raise ValueError(f'Unsupported mesh representation method:{mode}')


def stl2element_graph(mesh):
    """
    Represent STL mesh file as graph with mesh elements as nodes and adjacency relationships between elements as connections; implementation principle similar to grd2element_graph
    Node features: coordinates of three points of mesh element (3*3), normal vector (3)
    :param mesh:Mesh class instance in stl library
    :return: A_surface：coo_matrix；X_surface：numpy.narray
    """
    v0 = pandas.DataFrame(mesh.v0)
    v1 = pandas.DataFrame(mesh.v1)
    v2 = pandas.DataFrame(mesh.v2)

    v = pandas.concat([v0, v1, v2])
    v_duplicate = v.groupby(v.columns.to_list()).apply(
        lambda x: tuple(sorted(set(list(x.index)))) if len(set(x.index)) > 1 else None).to_list()
    duplicate = [x for x in v_duplicate if x]

    # %%
    A_surface = create_As(duplicate, v0.shape[0])
    A_surface[A_surface == 1] = 0
    A_surface[A_surface == 2] = 1
    X_surface = pandas.DataFrame(np.concatenate([mesh.points, mesh.normals], axis=1))
    return X_surface.values, A_surface.tocoo()


def stl2vertex_graph(mesh):
    """
    Represent STL mesh file as graph with mesh nodes as graph vertices and adjacency relationships between nodes as connections; implementation principle similar to grd2vertex_graph
    :param mesh: stl mesh object
    :return: X_point(array),A_point(coo_matrix)
    """
    row = np.arange(len(mesh.v0))
    row = np.concatenate([row, row, row + len(mesh.v0)])
    col = row + len(mesh.v0)
    col[len(mesh.v0):2 * len(mesh.v0)] += len(mesh.v0)
    A_point = coo_matrix((np.ones(len(col)), (row, col)), shape=(3 * len(mesh.v0), 3 * len(mesh.v0)),
                         dtype=np.int8).tocsr()
    A_point += A_point.transpose()

    # Adjacency matrix within triangle
    X_point = pandas.DataFrame(np.concatenate([mesh.v0, mesh.v1, mesh.v2], axis=0))

    points_duplicate = X_point.groupby(X_point.columns.to_list()).apply(
        lambda x: list(x.index) if len(x.index) > 1 else None).to_list()
    points_duplicate = [x for x in points_duplicate if x]
    A_adjacent_duplicated = create_As(points_duplicate, X_point.shape[0])

    A_ = A_point.dot(A_adjacent_duplicated)
    A_[A_ > 1] = 1
    A_ = A_adjacent_duplicated.dot(A_)
    A_[A_ > 1] = 1
    drop_index_row = [j for i in points_duplicate for j in i[1:]]
    drop_index_col = [j for i in points_duplicate for j in i[1:]]
    remain_index_row = sorted(list(set(range(A_.shape[0])) - set(drop_index_row)))
    remain_index_col = sorted(list(set(range(A_.shape[0])) - set(drop_index_col)))
    final_A = A_[remain_index_row, :][:, remain_index_col]

    A_point = final_A
    X_point = X_point.drop(index=drop_index_row)

    return X_point.values, A_point.tocoo()


def preprocess_stl_meshes(input_path, output_path, mode=None):
    """
    Generate graph-based stl 3D unstructured mesh representation
    :param input_path: Mesh file path
    :param output_path: Output file storage path
    :param mode: E, element-based graph representation; V, node-based graph representation
    """
    input_path = pathlib.Path(input_path)
    output_path = pathlib.Path(output_path)
    if not output_path.exists():
        output_path.mkdir()

    if mode == 'V':
        output_path = output_path / 'vertex_graph'
        if not output_path.exists():
            output_path.mkdir()
    elif mode == 'E':
        output_path = output_path / 'element_graph'
        if not output_path.exists():
            output_path.mkdir()

    for file in input_path.iterdir():
        if '.stl' in file.suffix:
            print(f'Processing STL file {file}.')
            x_path = output_path / (file.stem + '.npy')
            a_path = output_path / (file.stem + '.npz')
            if x_path.exists() and a_path.exists():
                continue
            #mesh = stl.mesh.Mesh.from_file(file)
            mesh = "null"
            if mode == 'V':
                X, A = stl2vertex_graph(mesh)
            elif mode in [None, 'E']:
                X, A = stl2element_graph(mesh)
            else:
                raise ValueError(f'Unsupported mesh representation method:{mode}')
            np.save(x_path, X)
            print(X.shape)
            save_npz(a_path, A)


# Only import partial modules, not underlying implementation
__all__ = ['preprocess_grd_meshes', 'grd2vertex_graph', 'grd2element_graph', 'stl2element_graph', 'stl2vertex_graph',
           'preprocess_stl_meshes']
