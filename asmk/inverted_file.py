"""Inverted file implementations for efficient search in a set of visual words"""

import numpy as np

from . import io_helpers


class IVF:
    """Inverted File for efficient feature indexation with idf support. Can be updated."""

    def __init__(self, norm_factor, n_images, ivf_vecs, ivf_image_ids, counts, idf, imid_offset, *, use_idf):
        self.params = {
            "use_idf": use_idf,
        }

        self.norm_factor = norm_factor
        self.n_images = n_images
        self.ivf_vecs = ivf_vecs
        self.ivf_image_ids = ivf_image_ids
        self.counts = counts
        self.idf = idf
        self.imid_offset = imid_offset


    @classmethod
    def initialize_empty(cls, *, codebook_size, **params):
        """Return an empty IVF object given codebook params (size) and IVF params."""
        ivf_vecs = [None for _ in range(codebook_size)]
        ivf_image_ids = [None for _ in range(codebook_size)]
        counts = np.zeros(codebook_size, dtype=np.int32)
        idf = np.ones(counts.shape, dtype=np.float32)

        return cls(**params, norm_factor=[], n_images=0, ivf_vecs=ivf_vecs,
                   ivf_image_ids=ivf_image_ids, counts=counts, idf=idf, imid_offset=0)

    #
    # Index and search
    #

    @staticmethod
    def _append_to_np_array(arr, size, item):
        initial_size, increase_ratio = 10, 1.5

        if arr is None:
            # Initialization
            arr = np.zeros((initial_size,) + item.shape, dtype=item.dtype)
        elif size >= arr.shape[0]:
            # Extension
            new_size = int(np.ceil(arr.shape[0] * increase_ratio))
            arr = np.resize(arr, (new_size,) + arr.shape[1:])

        arr[size] = item
        return arr


    def add(self, des, word_ids, image_ids, *, progress=None):
        """Add descriptors with corresponding visual word ids and image ids to this ivf"""
        image_ids += self.imid_offset
        uniq_image_ids = np.unique(image_ids)
        assert uniq_image_ids.min() >= self.n_images # The next chunk must be consequtive

        norm_append = np.zeros(uniq_image_ids.max() + 1 - len(self.norm_factor))
        self.norm_factor = np.concatenate((self.norm_factor, norm_append))
        self.n_images = np.max((self.n_images, uniq_image_ids.max() + 1))

        for i, word in io_helpers.progress(enumerate(word_ids), size=len(word_ids),
                                           frequency=progress, header="Index"):
            self.ivf_vecs[word] = self._append_to_np_array(self.ivf_vecs[word], self.counts[word],
                                                           des[i])
            self.ivf_image_ids[word] = self._append_to_np_array(self.ivf_image_ids[word],
                                                                self.counts[word], image_ids[i])
            self.counts[word] += 1

            if self.params["use_idf"]:
                self.idf[i] = np.log(self.n_images / self.counts[i])

                norm_delta = self.idf[i]**2 # New norm for vw
                norm_delta -= np.log(self.n_images / (self.counts[i]-1))**2 # Old norm for vw
                self.norm_factor[image_ids[i]] += norm_delta
            else:
                self.norm_factor[image_ids[i]] += 1


    def search(self, des, word_ids, *, similarity_func, topk):
        """Search in this ivf with given descriptors and corresponding visual word ids. Return
            similarity computed by provided function downweighted by idf and accumulated for all
            visual words. Return topk results per query."""
        scores = np.zeros(self.n_images)
        q_norm_factor = 0

        for qvec, word in zip(des, word_ids):
            q_norm_factor += self.idf[word]**2
            if self.ivf_image_ids[word] is None:
                # Empty visual word
                continue

            image_ids, sim = similarity_func(qvec, self.ivf_vecs[word][:self.counts[word]],
                                             self.ivf_image_ids[word][:self.counts[word]])

            sim *= (self.idf[word]**2) # apply idf
            sim /= np.sqrt(self.norm_factor[image_ids]) # normalize
            scores[image_ids] += sim

        scores = scores / np.sqrt(q_norm_factor)
        ranks = np.argsort(-scores)[:topk]
        return ranks - self.imid_offset, scores[ranks]

    #
    # Load, save and stats
    #

    @property
    def stats(self):
        """Return a shallow dictionary with stats of the ivf"""
        sum_counts = self.counts.sum()
        imbalance_factor = self.counts.shape[0] * np.power(self.counts, 2).sum() / sum_counts**2
        return {
            "vectors_per_image": sum_counts / self.n_images,
            "mean_entries_per_vw": self.counts.mean(),
            "empty_vw": sum(1 for x in self.counts if x == 0),
            "min_entries_per_vw": self.counts.min(),
            "max_entries_per_vw": self.counts.max(),
            "std_of_entries_per_vw": self.counts.std(),
            "imbalance_factor_of_vw": imbalance_factor,
        }


    def state_dict(self):
        """Return state dict which is a checkpoint of current state for future recovery"""
        return {
            "type": self.__class__.__name__,
            "params": self.params,
            "state": {
                "norm_factor": self.norm_factor,
                "n_images": self.n_images,
                "ivf_vecs": self.ivf_vecs,
                "ivf_image_ids": self.ivf_image_ids,
                "counts": self.counts,
                "idf": self.idf,
                "imid_offset": self.imid_offset,
            }
        }

    @classmethod
    def initialize_from_state(cls, state):
        """Initialize from a previously stored state_dict given an index factory"""
        assert state["type"] == cls.__name__
        if "imid_offset" not in state['state']:
            # For backwards compatibility
            state['state']['imid_offset'] = 0
        return cls(**state["params"], **state["state"])
