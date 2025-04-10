"""
Abstract evaluator to control how (batched) objective evaluations are performed 

Authors:    Tucker Hartland <hartland1@llnl.gov>
            Nai-Yuan Chiang <chiang7@llnl.gov>
"""


class Evaluator(object):
    """
    An interface for evaluation of a function at x points (nsamples of dimension nx).
    User can derive this interface and override the run() method to implement custom multiprocessing.
    """

    def run(self, fun, x):
        """
        Evaluates fun at x.

        Parameters
        ---------
        fun : function to evaluate: (nsamples, nx) -> (nsample, 1)

        x : np.ndarray[nsamples, nx]
            nsamples points of nx dimensions.

        Returns
        -------
        np.ndarray[nsample, 1]
            fun evaluations at the nsamples points.

        """
        return fun(x)
