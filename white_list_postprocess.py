import numpy as np
import argparse

if __name__ == '__main__':
    """
    EXAMPLE
    usage: python white_list_postprocess.py --input_file wiki-Vote_mapped_trustrank_white_list.txt --mapping_file wiki-Vote_map.txt
    generates file: wiki-Vote_trustrank_white_list.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str,
                    help="input file (txt format)")
    ap.add_argument("-m", "--mapping_file", type=str,
                    help="mapping file (txt format)")
    args = ap.parse_args()

    input_file = args.input_file
    white_list = np.loadtxt(input_file, dtype=np.int32)
    map_file = args.mapping_file
    map = np.loadtxt(map_file, dtype=np.int32)

    vertices = map[:, 0]
    ids = map[:, 1]
    map_table = dict(zip(ids, vertices))

    white_list_raw = np.vectorize(lambda x: map_table[x])(white_list)  # convert to raw

    file_name = input_file.replace('_mapped', '')
    np.savetxt(file_name, white_list_raw, fmt='%1u')

    print("Postprocess completed")
