import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.utils.btree import _BTree


def get_censoring_dist(times, event_observed):
    # _dataset = train_dataset.dataset
    # times, event_observed = [d['time_at_event'] for d in _dataset], [d['y'] for d in _dataset]
    # times, event_observed = [d for d in times], [d for d in event_observed]
    times = list(times)
    all_observed_times = set(list(times))
    kmf = KaplanMeierFitter()
    kmf.fit(times, event_observed)

    censoring_dist = {time: kmf.predict(time) for time in all_observed_times}
    # Prevent zero censoring probability at any time point
    censoring_dist = {
        time: max(prob, 1e-6) for time, prob in censoring_dist.items()
    }  # Prevent zero censoring

    return censoring_dist


def concordance_index_ipcw(event_times, predictions, event_observed, censoring_dist):
    """
    Calculates Uno's C-index, weighted by IPCW and using time-dependent scores.
    Parameters
    ----------
    event_times : iterable
        Observed survival times.
    predictions : ndarray
        Time-dependent predicted scores (shape: [n_samples, n_timepoints]).
    event_observed : iterable
        Event indicator: 1 if event occurred, 0 if censored.
    censoring_dist : dict
        Dictionary of survival probabilities for censoring at each time point.
    Returns
    -------
    c-index : float
        Uno's C-index.
    """
    predicted_scores = 1 - np.asarray(predictions, dtype=float)
    event_times = np.asarray(event_times, dtype=float)
    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        event_observed = np.asarray(event_observed, dtype=float).ravel()
        if event_observed.shape != event_times.shape:
            raise ValueError(
                "Observed events must be 1-dimensional of same length as event times"
            )

    num_correct, num_tied, num_pairs = _concordance_summary_statistics(
        event_times, predicted_scores, event_observed, censoring_dist
    )

    return _concordance_ratio(num_correct, num_tied, num_pairs)


def _concordance_ratio(num_correct, num_tied, num_pairs):
    if num_pairs == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return (num_correct + num_tied / 2) / num_pairs


def _concordance_summary_statistics(
    event_times, predicted_event_times, event_observed, censoring_dist
):  # pylint: disable=too-many-locals
    """Find the concordance index in n * log(n) time.
    Assumes the data has been verified by lifelines.utils.concordance_index first.
    """
    # Here's how this works.
    #
    # It would be pretty easy to do if we had no censored data and no ties. There, the basic idea
    # would be to iterate over the cases in order of their true event time (from least to greatest),
    # while keeping track of a pool of *predicted* event times for all cases previously seen (= all
    # cases that we know should be ranked lower than the case we're looking at currently).
    #
    # If the pool has O(log n) insert and O(log n) RANK (i.e., "how many things in the pool have
    # value less than x"), then the following algorithm is n log n:
    #
    # Sort the times and predictions by time, increasing
    # n_pairs, n_correct := 0
    # pool := {}
    # for each prediction p:
    #     n_pairs += len(pool)
    #     n_correct += rank(pool, p)
    #     add p to pool
    #
    # There are three complications: tied ground truth values, tied predictions, and censored
    # observations.
    #
    # - To handle tied true event times, we modify the inner loop to work in *batches* of observations
    # p_1, ..., p_n whose true event times are tied, and then add them all to the pool
    # simultaneously at the end.
    #
    # - To handle tied predictions, which should each count for 0.5, we switch to
    #     n_correct += min_rank(pool, p)
    #     n_tied += count(pool, p)
    #
    # - To handle censored observations, we handle each batch of tied, censored observations just
    # after the batch of observations that died at the same time (since those censored observations
    # are comparable all the observations that died at the same time or previously). However, we do
    # NOT add them to the pool at the end, because they are NOT comparable with any observations
    # that leave the study afterward--whether or not those observations get censored.
    if np.logical_not(event_observed).all():
        return (0, 0, 0)

    observed_times = set(event_times)

    died_mask = event_observed.astype(bool)
    # TODO: is event_times already sorted? That would be nice...
    died_truth = event_times[died_mask]
    ix = np.argsort(died_truth)
    died_truth = died_truth[ix]

    died_pred = predicted_event_times[died_mask][ix]

    censored_truth = event_times[~died_mask]
    ix = np.argsort(censored_truth)
    censored_truth = censored_truth[ix]
    censored_pred = predicted_event_times[~died_mask][ix]

    censored_ix = 0
    died_ix = 0
    times_to_compare = {}
    for time in observed_times:
        times_to_compare[time] = _BTree(np.unique(died_pred[:, int(time)]))
    num_pairs = np.int64(0)
    num_correct = np.int64(0)
    num_tied = np.int64(0)

    # we iterate through cases sorted by exit time:
    # - First, all cases that died at time t0. We add these to the sortedlist of died times.
    # - Then, all cases that were censored at time t0. We DON'T add these since they are NOT
    #   comparable to subsequent elements.
    while True:
        has_more_censored = censored_ix < len(censored_truth)
        has_more_died = died_ix < len(died_truth)
        # Should we look at some censored indices next, or died indices?
        if has_more_censored and (
            not has_more_died or died_truth[died_ix] > censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                censored_truth,
                censored_pred,
                censored_ix,
                times_to_compare,
                censoring_dist,
            )
            censored_ix = next_ix
        elif has_more_died and (
            not has_more_censored or died_truth[died_ix] <= censored_truth[censored_ix]
        ):
            pairs, correct, tied, next_ix, weight = _handle_pairs(
                died_truth, died_pred, died_ix, times_to_compare, censoring_dist
            )
            for pred in died_pred[died_ix:next_ix]:
                for time in observed_times:
                    times_to_compare[time].insert(pred[int(time)])
            died_ix = next_ix
        else:
            assert not (has_more_died or has_more_censored)
            break

        num_pairs += pairs * weight
        num_correct += correct * weight
        num_tied += tied * weight

    return (num_correct, num_tied, num_pairs)


def _handle_pairs(truth, pred, first_ix, times_to_compare, censoring_dist):
    next_ix = first_ix
    truth_time = truth[first_ix]
    weight = 1.0 / (censoring_dist[truth_time] ** 2)
    while next_ix < len(truth) and truth[next_ix] == truth[first_ix]:
        next_ix += 1
    pairs = len(times_to_compare[truth_time]) * (next_ix - first_ix)
    correct = np.int64(0)
    tied = np.int64(0)
    for i in range(first_ix, next_ix):
        rank, count = times_to_compare[truth_time].rank(pred[i][int(truth_time)])
        correct += rank
        tied += count

    return (pairs, correct, tied, next_ix, weight)
