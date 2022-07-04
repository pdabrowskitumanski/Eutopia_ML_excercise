from topoly import alexander, generate_loop, Closure, writhe, Graph
import pandas as pd
import sys


"""
Usage: python build_dataset.py n where n is the number of structures in the dataset. 
Default number of structures is 500
"""
if len(sys.argv) < 2:
    loop_number = 500
else:
    loop_number = int(sys.argv[1])


def get_loop_props(loop_coords: list, loop_topology: str) -> dict:
    g = Graph(loop_coords)
    g.closed = True
    code = g.pdcode
    crossings = len(code.split(';')) - 1
    loop_writhe = 0 if crossings == 0 else writhe(loop_coords, closure=Closure.CLOSED)
    data_dict = {
        'writhe': loop_writhe,
        'crossings': crossings,
        'topology': loop_topology,
        'pdcode': code,
        'loop_coords': loop_coords,
    }
    return data_dict


loops = []
while len(loops) < loop_number:
    loop = generate_loop(100, 1, output='list')[0]
    topology = alexander(loop, closure=Closure.CLOSED)
    if topology == 'TooManyCrossings':
        continue
    loops.append(get_loop_props(loop, topology))
    print(len(loops) / loop_number, end='\r')


df = pd.DataFrame(loops)
df.to_csv('knots_dataset.csv', index=False)
