import numpy as np
import argparse
import os

if __name__ == '__main__':
    """
    EXAMPLE
    usage: python preprocess.py --input_file wiki-Vote.txt
    generates file: wiki-Vote_map.txt
    generates file: wiki-Vote_mapped.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--input_file", type=str,
                    help="input file (txt format)")
    ap.add_argument("-u", "--undirected", default=False, action='store_true',
                    help="if True, make the graph undirected")
    ap.add_argument("-i", "--input_folder", type=str, default=None,
                    help="input folder (None = search in current directory)")
    args = ap.parse_args()

    input_file_name = args.input_file
    if args.input_folder is not None:
        input_file_name = os.path.join(args.input_folder, input_file_name)
    edges = np.loadtxt(input_file_name, dtype=np.int32)
    file_name = args.input_file

    # create indices-IDs map table
    vertices = np.unique(edges.flatten())
    ids = np.arange(len(vertices))
    map = np.vstack([vertices, ids]).T
    np.savetxt(file_name.split(".")[0] + "_map.txt", map, fmt='%1u')

    # map raw indices to IDs
    map_table = dict(zip(vertices, ids))
    if args.undirected:
        reversed_edges = edges[:, [1, 0]]  # swap first and second row
        edges = np.vstack([edges, reversed_edges])
    edges = np.vectorize(lambda x: map_table[x])(edges)  # convert to ID
    np.savetxt(file_name.split(".")[0] + "_mapped.txt", edges, fmt='%1u')

    print("Preprocess completed")
