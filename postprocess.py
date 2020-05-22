import numpy as np
import argparse

if __name__ == '__main__':
    """
    EXAMPLE
    usage: python postprocess.py --input_file wiki-Vote_mapped_pagerank_top-5.txt --mapping_file wiki-Vote_map.txt
    generates file: wiki-Vote_pagerank_top-5.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str,
                    help="input file (txt format)")
    ap.add_argument("-m", "--mapping_file", type=str,
                    help="mapping file (txt format)")
    args = ap.parse_args()

    input_file = args.input_file
    rank = np.loadtxt(input_file, dtype=np.int32)
    map_file = args.mapping_file
    map = np.loadtxt(map_file, dtype=np.int32)

    vertices = map[:, 0]
    ids = map[:, 1]
    map_table = dict(zip(ids, vertices))

    ranked_vertices_id = rank[:, 0]
    ranked_vertices_raw = np.vectorize(lambda x: map_table[x])(ranked_vertices_id)  # convert to raw
    rank[:, 0] = ranked_vertices_raw

    file_name = input_file.replace('_mapped', '')
    np.savetxt(file_name, rank, fmt='%1u')

    print("Postprocess completed")
