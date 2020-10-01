"""A wrapper around all asmk-related packages for convenient use"""

import os.path
import time
import numpy as np

from . import kernel as kern_pkg, codebook as cdb_pkg, index as idx_pkg, inverted_file as ivf_pkg
from . import io_helpers


class ASMKMethod:
    """Class to keep necessary objects and provide easy access to asmk method's steps. Each step
        of asmk method corresponds to one method."""

    def __init__(self, params, metadata, *, codebook=None, kernel=None, inverted_file=None):
        self.params = params
        self.metadata = metadata

        self.codebook = codebook
        self.kernel = kernel
        self.inverted_file = inverted_file


    @classmethod
    def initialize_untrained(cls, params):
        """Initialize asmk method before training"""
        return cls(params, {})

    #
    # Method steps
    #

    def train_codebook(self, vecs, *, cache_path=None, step_params=None):
        """The first step of the method - training codebook (or loading from cache). Return new
            ASMKMethod object, do not change self."""
        assert not self.codebook, "Codebook already trained"
        index_factory = idx_pkg.initialize_faiss_index(**self.params['index'])
        step_params = step_params or self.params.get("train_codebook")

        if cache_path and os.path.exists(cache_path):
            time0 = time.time()
            cdb = cdb_pkg.Codebook.initialize_from_state(io_helpers.load_pickle(cache_path),
                                                         index_factory=index_factory)
            cdb.index()
            assert cdb.params == step_params['codebook']
            metadata = {"load_time": time.time() - time0}
        else:
            cdb = cdb_pkg.Codebook(**step_params['codebook'], index_factory=index_factory)
            metadata = cdb.train(vecs)
            if cache_path:
                io_helpers.save_pickle(cache_path, cdb.state_dict())

        metadata["index_class"] = index_factory.__class__.__name__
        return self.__class__({**self.params, "train_codebook": step_params},
                              {**self.metadata, "train_codebook": metadata},
                              codebook=cdb)


    def build_ivf(self, vecs, imids, *, cache_path=None, step_params=None):
        """The second step of the method - building the ivf (or loading from cache). Return new
            ASMKMethod object, do not change self."""
        assert not self.kernel and not self.inverted_file, "Inverted file already built"
        step_params = step_params or self.params.get("build_ivf")
        kern = kern_pkg.ASMKKernel(self.codebook, **step_params['kernel'])

        if cache_path and os.path.exists(cache_path):
            time0 = time.time()
            ivf = ivf_pkg.IVF.initialize_from_state(io_helpers.load_pickle(cache_path))
            metadata = {"load_time": time.time() - time0}
        else:
            ivf = ivf_pkg.IVF.initialize_empty(**step_params['ivf'],
                                               codebook_params=self.codebook.params)

            time0 = time.time()
            quantized = self.codebook.quantize(vecs, imids, **step_params["quantize"])
            aggregated = kern.aggregate(*quantized, **step_params["aggregate"])
            ivf.add(*aggregated)
            metadata = {"index_time": time.time() - time0}

            if cache_path:
                io_helpers.save_pickle(cache_path, ivf.state_dict())

        metadata['ivf_stats'] = ivf.stats
        return self.__class__({**self.params, "build_ivf": step_params},
                              {**self.metadata, "build_ivf": metadata},
                              codebook=self.codebook, kernel=kern, inverted_file=ivf)


    def query_ivf(self, qvecs, qimids, *, step_params=None):
        """The last step of the method - querying the ivf. Return tuple (metadata, images, ranks,
            scores), do not change self."""

        step_params = step_params or self.params.get("query_ivf")

        time0 = time.time()
        images, ranks, scores = self.accumulate_scores(self.codebook, self.kernel, \
                                            self.inverted_file, qvecs, qimids, step_params)
        metadata = {"query_avg_time": (time.time()-time0)/len(ranks)}
        return metadata, images, ranks, scores


    #
    # Helper functions
    #

    @staticmethod
    def accumulate_scores(cdb, kern, ivf, qvecs, qimids, params):
        """Accumulate scores for every query image (qvecs, qimids) given codebook, kernel,
            inverted_file and parameters."""
        similarity_func = lambda *x: kern.similarity(*x, **params["similarity"])

        imids_all, ranks_all, scores_all = [], [], []
        for i in range(qimids.min(), qimids.max()+1):
            qdes = qvecs[qimids==i]
            quantized = cdb.quantize(qdes, **params["quantize"])
            aggregated = kern.aggregate(*quantized, **params["aggregate"])
            scores = ivf.search(*aggregated, **params["search"], similarity_func=similarity_func)
            ranks = np.argsort(-scores)
            imids_all.append(i)
            ranks_all.append(ranks)
            scores_all.append(scores[ranks])

        return np.array(imids_all), np.vstack(ranks_all), np.vstack(scores_all)
