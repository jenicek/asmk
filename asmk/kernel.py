"""Kernel functionality implementation - aggregation and similarity computation"""

import numpy as np

from . import hamming


class ASMKKernel:
    """Kernel for ASMK with the option of binarization."""

    def __init__(self, codebook, *, binary):
        self.params = {
            "binary": binary,
        }

        self.codebook = codebook

    #
    # Aggregation
    #

    def _aggregate_image(self, des, word_ids):
        """Aggregate descriptors (with corresponding visual word ids) for a single image"""
        unique_ids = np.unique(word_ids)

        if self.params["binary"]:
            # Storage for packed booleans
            ades = np.empty((unique_ids.shape[0], int(np.ceil(des.shape[1] / 32))), dtype=np.uint32)
        else:
            ades = np.empty((unique_ids.shape[0], des.shape[1]), dtype=np.float32)

        for i, word in enumerate(unique_ids):
            residuals = des[word_ids==word] - self.codebook.centroids[word]
            residual = residuals.sum(0)

            if self.params["binary"]:
                ades[i] = hamming.binarize_and_pack(residual.astype(np.float32))
            else:
                ades[i] = normalize_vec_l2(np.expand_dims(residual, axis=0)).squeeze()

        return ades, unique_ids

    def aggregate(self, des, word_ids, image_ids=None, **kwargs):
        """Aggregate descriptors with corresponding visual word ids for corresponding image ids.
            When image_ids is None, act as if all descriptors come from the same image."""
        if image_ids is None:
            return self._aggregate_image(des, word_ids)

        agg_des, agg_word_ids, agg_image_ids = [], [], []
        for image_id in np.unique(image_ids):
            mask = (image_ids == image_id)
            agg_des_incr, agg_word_ids_incr = \
                    self._aggregate_image(des[mask], word_ids[mask], **kwargs)
            agg_des.append(agg_des_incr)
            agg_word_ids.append(agg_word_ids_incr)
            agg_image_ids.append(np.full(agg_word_ids_incr.shape[0], image_id))

        agg_des  = np.vstack(agg_des)
        agg_word_ids = np.hstack(agg_word_ids)
        agg_image_ids = np.hstack(agg_image_ids)

        return agg_des, agg_word_ids, agg_image_ids

    #
    # Similarity
    #

    def similarity(self, qvec, vecs, image_ids, *, alpha, similarity_threshold):
        """Compute similarity between given query vector and database feature vectors with their
            corresponding image ids. Alpha is the similarity exponent after the similarity
            threshold is applied."""
        # Compute similarity with vw residuals for all other images
        if self.params["binary"]:
            norm_hdist = hamming.hamming_cdist_packed(qvec.reshape(1, -1), vecs)
            sim = -2*norm_hdist.squeeze(0) + 1 # normalized hamming dist -> similarity in [-1, 1]
        else:
            sim = np.matmul(vecs, qvec)

        # Compute vw scores
        mask = (sim>=similarity_threshold)
        sim = np.power(sim[mask], alpha) # monomial kernel
        return image_ids[mask], sim

    #
    # Load and save
    #

    def state_dict(self):
        """Return state dict which is a checkpoint of current state for future recovery"""
        return {
            "type": self.__class__.__name__,
            "params": self.params,
        }

    @classmethod
    def initialize_from_state(cls, state, codebook):
        """Initialize from a previously stored state_dict given a codebook"""
        assert state["type"] == cls.__name__
        return cls(**state["params"], codebook=codebook)


# Helper functions

def normalize_vec_l2(vecs):
    """Perform l2 normalization on each vector in a given matrix (axis 1)"""
    norm = np.linalg.norm(vecs, ord=2, axis=1, keepdims=True) + 1e-6
    return vecs / norm
