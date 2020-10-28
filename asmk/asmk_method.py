"""A wrapper around all asmk-related packages for convenient use"""

import os.path
import time
import numpy as np

from . import kernel as kern_pkg, codebook as cdb_pkg, index as idx_pkg, inverted_file as ivf_pkg
from . import io_helpers


class ASMKMethod:
    """
    Class to keep necessary objects and provide easy access to asmk method's steps. Each step
    of asmk method corresponds to one method. Use initialize_untrained() class method instead
    of directly calling the constructor.

    :param dict params: contains keys index, train_codebook, build_ivf and query_ivf, each
        containing the corresponding step parameters
    :param dict metadata: only stored by this object, never changed
    :param Codebook codebook: object from the codebook module
    :param ASMKKernel kernel: object from the kernel module
    :param IVF inverted_file: object from the inverted_file module
    """

    def __init__(self, params, metadata, *, codebook=None, kernel=None, inverted_file=None):
        self.params = params
        self.metadata = metadata

        self.codebook = codebook
        self.kernel = kernel
        self.inverted_file = inverted_file


    @classmethod
    def initialize_untrained(cls, params):
        """Initialize asmk method before training, provided only params (see constructor docs)"""
        return cls(params, {})

    #
    # Method steps
    #

    def train_codebook(self, vecs, *, cache_path=None, step_params=None):
        """The first step of the method - training the codebook (or loading from cache)

        :param ndarray vecs: 2D numpy array, rows are vectors for codebook training
        :param str cache_path: trained codebook will be stored under given file path and loaded
            next time without training (None to turn off)
        :param dict step_params: parameters that will override stored parameters for this step
            (self.params['train_codebook'])
        :return: new ASMKMethod object (containing metadata of this step), do not change self
        """
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
        """The second step of the method - building the ivf (or loading from cache)

        :param ndarray vecs: 2D numpy array, rows are vectors to be indexed by the ivf
        :param ndarray imids: 1D numpy array of image ids corresponding to 'vecs'
        :param str cache_path: built ivf will be stored under given file path and loaded
            next time without training (None to turn off)
        :param dict step_params: parameters that will override stored parameters for this step
            (self.params['build_ivf'])
        :return: new ASMKMethod object (containing metadata of this step), do not change self
        """

        builder = self.create_ivf_builder(cache_path=cache_path, step_params=step_params)

        if not builder.loaded_from_cache:
            # Skip if loaded, otherwise add everything at once
            builder.add(vecs, imids)

        return self.add_ivf_builder(builder)


    def create_ivf_builder(self, *, cache_path=None, step_params=None):
        """Part of the second step of the method, see build_ivf() method implementation for usage

        :param str cache_path: built ivf will be stored under given file path and loaded
            next time without training (None to turn off)
        :param dict step_params: parameters that will override stored parameters for this step
            (self.params['build_ivf'])
        :return: IvfBuilder object
        """
        assert not self.kernel and not self.inverted_file, "Inverted file already built"
        step_params = step_params or self.params.get("build_ivf")
        kern = kern_pkg.ASMKKernel(self.codebook, **step_params['kernel'])

        return IvfBuilder(step_params, self.codebook, kern, cache_path=cache_path)


    def add_ivf_builder(self, ivf_builder):
        """Part of the second step of the method, see build_ivf() method implementation for usage

        :param IvfBuilder ivf_builder: Builder with vectors added
        :return: new ASMKMethod object (containing metadata from the builder), do not change self
        """
        ivf_metadata = ivf_builder.save()

        return self.__class__({**self.params, "build_ivf": ivf_builder.step_params},
                              {**self.metadata, "build_ivf": ivf_metadata},
                              codebook=self.codebook, kernel=ivf_builder.kernel,
                              inverted_file=ivf_builder.ivf)


    def query_ivf(self, qvecs, qimids, *, step_params=None):
        """The last step of the method - querying the ivf

        :param ndarray qvecs: 2D numpy array, rows are vectors, each acting as a query for the ivf
        :param ndarray qimids: 1D numpy array of image ids corresponding to 'qvecs'
        :param dict step_params: parameters that will override stored parameters for this step
            (self.params['query_ivf'])
        :return: tuple (dict metadata, ndarray images, 2D ndarray ranks, 2D ndarray scores), do not
            change self
        """

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


class IvfBuilder:
    """Inverted file (IVF) wrapper simplifying vector addition

    :param dict step_params: contains parameters for build_ivf step
    :param Codebook codebook: object from the codebook module
    :param ASMKKernel kernel: object from the kernel module
    :param str cache_path: built ivf will be stored under given file path and loaded
        next time without training (None to turn off)
    """

    def __init__(self, step_params, codebook, kernel, *, cache_path):
        self.step_params = step_params
        self.codebook = codebook
        self.kernel = kernel

        if cache_path and os.path.exists(cache_path):
            time0 = time.time()
            self.ivf = ivf_pkg.IVF.initialize_from_state(io_helpers.load_pickle(cache_path))
            self.metadata = {"load_time": time.time() - time0}
            self.cache_path = None
        else:
            self.ivf = ivf_pkg.IVF.initialize_empty(**step_params['ivf'],
                                                    codebook_params=codebook.params)
            self.metadata = {"index_time": 0}
            self.cache_path = cache_path

    @property
    def loaded_from_cache(self):
        """If the contained IVF was loaded (otherwise, it is empty after initialization)"""
        return "load_time" in self.metadata

    def add(self, vecs, imids):
        """Add descriptors and cooresponding image ids to the IVF

        :param np.ndarray vecs: 2D array of local descriptors
        :param np.ndarray imids: 1D array of image ids
        """
        time0 = time.time()
        quantized = self.codebook.quantize(vecs, imids, **self.step_params["quantize"])
        aggregated = self.kernel.aggregate(*quantized, **self.step_params["aggregate"])
        self.ivf.add(*aggregated)
        self.metadata['index_time'] += time.time() - time0

    def save(self):
        """Save to cache path if defined

        :return: dict metadata with ivf stats
        """
        if self.cache_path:
            io_helpers.save_pickle(self.cache_path, self.ivf.state_dict())

        return {**self.metadata, "ivf_stats": self.ivf.stats}
