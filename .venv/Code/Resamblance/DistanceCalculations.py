import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.special import kl_div
from scipy.stats import wasserstein_distance


class cos_distance:
    # Description: Cosine of the angle between two vectors in n-dimensional space
    # Interpretation: The lower the value, the higher the resamblance between two sets of values will be -> Threshold of 0.3 indicates that STD variables resamblances the RD variable
    # r: Attribute of real data
    # s: Attribute of synthetic data
    def cos_distance(r, s):
        r, s = np.array(r), np.array(s)
        dot_product = np.dot(r, s)
        norm_r = np.linalg.norm(r)
        norm_s = np.linalg.norm(s)
        return 1 - (dot_product / (norm_r * norm_s))

class jensen_shannon_distance:

    def jensen_shannon_distance(self, p, q, base=2):
        # Description: Jensen-Shannon-Distance measures the similarity between two distributions
        # Interpretation: A value lower than 0.1 implies perfect resemblance since higher values would indicate that the difference in distributions are higher
        # p: probability distribution of RD attribute
        # q: probability distribution of SD attribute
        # m: pointwise mean of p and q
        # D: Kullback-Leibler-Distance
        p = np.asarray(p)
        q = np.asarray(q)

        p /= p.sum()
        q /= q.sum()


        return jensenshannon(p, q, base=base)

class kl_divergence:

    def kl_divergence(self, p, q):
        # Description:
        kl_divergence = kl_div(p, q)

        return kl_divergence


class wasserstein_distance:

    def wasserstein_distance(self, r, s):
        # Description: can be seen as the minimum cost required to transform a vector (r) into another vector (s), where the cost is measured
        # as the amount of distribution weight that must be moved, multiplied by the distance it has to be moved
        # Interpretation: A threshold of 0.3 is proposed to assure the resembalance of the attribute
        # r: cumulative distribution function of real data
        # s: cumulative distribution function of synthetic data

        wd = wasserstein_distance(r, s)

        return wd