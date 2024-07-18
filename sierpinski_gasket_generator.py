import numpy as np
import pandas as pd
import open3d as o3d
from noise import pnoise3
from multiprocessing import Pool

def midpoint(p1, p2):
    return (p1 + p2) / 2.0

def create_tetrahedrons(points):
    midpoints = [midpoint(points[i], points[j]) for i in range(4) for j in range(i+1, 4)]
    return [
        [points[0], midpoints[0], midpoints[1], midpoints[2]],
        [points[1], midpoints[0], midpoints[3], midpoints[4]],
        [points[2], midpoints[1], midpoints[3], midpoints[5]],
        [points[3], midpoints[2], midpoints[4], midpoints[5]]
    ]

def process_level(tetrahedrons):
    result = []
    for tetra in tetrahedrons:
        result.extend(create_tetrahedrons(np.array(tetra)))
    return result

def generate_sierpinski(level):
    points = np.array([
        [0.0, 0.0, 1.0],
        [np.sqrt(8/9), 0.0, -1/3],
        [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
        [-np.sqrt(2/9), -np.sqrt(2/3), -1/3]
    ])

    tetrahedrons = [points]
    for i in range(level):
        with Pool() as pool:
            tetrahedrons = pool.map(process_level, np.array_split(tetrahedrons, pool._processes))
            tetrahedrons = [item for sublist in tetrahedrons for item in sublist]
            print(f'Level {i+1} completed with {len(tetrahedrons)} tetrahedrons')

    return tetrahedrons

def save_to_disk(tetrahedrons):
    verts = np.array([vert for tetra in tetrahedrons for vert in tetra])
    faces = []
    for i in range(len(tetrahedrons)):
        base_index = i * 4
        faces.extend([
            [base_index, base_index+1, base_index+2],
            [base_index, base_index+1, base_index+3],
            [base_index, base_index+2, base_index+3],
            [base_index+1, base_index+2, base_index+3]
        ])
    faces = np.array(faces)
    
    vertex_colors = np.zeros((verts.shape[0], 3))
    for i in range(verts.shape[0]):
        x, y, z = verts[i]
        r = 0.5 + 0.5 * pnoise3(x, y, z)
        g = 0.5 + 0.5 * pnoise3(x + 100, y + 100, z + 100)
        b = 0.5 + 0.5 * pnoise3(x + 200, y + 200, z + 200)
        vertex_colors[i] = [r, g, b]

    verts = verts.astype(np.float64)
    vertex_colors = vertex_colors.astype(np.float64)

    np.set_printoptions(precision=15, suppress=True)

    data = np.hstack((verts, vertex_colors))
    df = pd.DataFrame(data, columns=['x', 'y', 'z', 'r', 'g', 'b'])

    chunk_size = 100000
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]
        chunk.to_csv(f'vertices_and_colors_{start}.csv', index=False, float_format='%.15f')

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

    o3d.io.write_triangle_mesh("mesh.gltf", mesh)

if __name__ == '__main__':
    level = 40  
    tetrahedrons = generate_sierpinski(level)
    save_to_disk(tetrahedrons)
