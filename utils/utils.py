import os
import json
import pickle
from itertools import combinations
from datetime import datetime
import math

from tqdm import tqdm
import random

from sklearn.metrics import adjusted_mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN

import networkx as nx
import networkx.algorithms.community as nx_comm
import cugraph as cnx
import cudf as gd

import markov_clustering as mc
import hdbscan


def get_cluster_assignment(cluster_type, cluster_params, corpus_embeddings):

    if cluster_type not in ["agglomerative", "HDBScan", "SLINK"]:
        raise ValueError('cluster_type must be "agglomerative", "HDBScan", "community" or "SLINK"')
    if cluster_type == "agglomerative":
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering linkage" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering linkage"')
        # if "metric" not in cluster_params:
        #     raise ValueError('cluster_params must contain "metric"')
    if cluster_type == "HDBScan":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "min samples" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
    if cluster_type == "SLINK":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering affinity" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering affinity"')

    if cluster_type == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_params["threshold"],
            linkage=cluster_params["clustering linkage"],
            metric=cluster_params["metric"] if "metric" in cluster_params else cluster_params["affinity"]
        )

    if cluster_type == "SLINK":
        clustering_model = DBSCAN(
            eps=cluster_params["threshold"],
            min_samples=cluster_params["min cluster size"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "HDBScan":
        clustering_model = hdbscan.HDBSCAN(
            min_cluster_size=cluster_params["min cluster size"],
            min_samples=cluster_params["min samples"],
            gen_min_span_tree=True
        )

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_


    return cluster_assignment