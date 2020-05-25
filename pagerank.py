from pyspark import SparkConf, SparkContext
import time
import numpy as np
import argparse


def parse_edge(e):
    """Splits an edge into source and destination"""
    source, dest = e.split(' ')
    return source, dest


def contribution(out_nodes, tot_rank):
    """Returns the contribution given from a source node to its out-nodes"""
    n_out_nodes = len(out_nodes)  # number of out-edges
    for out_node in out_nodes:
        yield (out_node, tot_rank / n_out_nodes)


if __name__ == '__main__':
    """
    EXAMPLE
    usage: python pagerank.py --input_file wiki-Vote_mapped.txt --num_iterations 20 --k_top 5
    python pagerank.py --input_file com-lj_mapped.txt --num_iterations 1 --k_top 5
    generates files: wiki-Vote_mapped_pagerank_top-5.txt
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input_file", type=str,
                    help="input file (txt format)")
    ap.add_argument("-n", "--num_iterations", type=int, default=20,
                    help="number of Pagerank iterations")
    ap.add_argument("-d", "--dumping_factor", type=float, default=0.8,
                    help="dumping factor")
    ap.add_argument("-k", "--k_top", type=int, default=0,
                    help="top k nodes to retrieve (0 = all nodes")
    args = ap.parse_args()

    # create spark context
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    lines = sc.textFile(args.input_file)

    # params
    n_iter = args.num_iterations
    d = args.dumping_factor
    top_k = args.k_top

    # initialize edges (source_id and list of out-nodes)
    nodes = lines.map(lambda x: parse_edge(x)).distinct().groupByKey()
    n_nodes = nodes.count()
    if not top_k:
        top_k = n_nodes
    # initialize ranks (source_id and rank)
    ranks = nodes.map(lambda x: (x[0], 1.0 / n_nodes))

    start = time.time()

    for i in range(n_iter):
        # calculates node contributions to the rank of other nodes (dest node and partial rank)
        contrib = nodes.join(ranks).flatMap(lambda x: contribution(x[1][0], x[1][1]))
        # update node ranks based on their in-nodes contributions
        ranks = contrib.reduceByKey(lambda x, y: x + y).mapValues(lambda x: x * d + (1. - d) / n_nodes)

    # sort nodes by rank and takes the k best ranks
    ranks = ranks.map(lambda x: (x[1], x[0])).sortByKey(False).map(lambda x: (x[1], x[0])).take(top_k)

    np_ranks = np.array(ranks).astype(float)
    fmt = '%1u', '%1.9f'
    if not args.k_top:
        top_k = "all"
    np.savetxt(args.input_file.split(".")[0] + "_pagerank_top-" + str(top_k) + ".txt", np_ranks, fmt=fmt)

    print("Total program time: %.2f seconds" % (time.time() - start))
    print("Top-{} nodes:".format(top_k), ranks[:5])
    sc.stop()
