import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def linear_interpolation(left, right, alpha):
    """
    Linear interpolation between `left` and `right`.
    :param left: (float) left boundary
    :param right: (float) right boundary
    :param alpha: (float) coeff in [0, 1]
    :return: (float)
    """

    return left + alpha * (right - left)

class PiecewiseSchedule(object):
    """
    Piecewise schedule.
    :param endpoints: ([(int, int)])
        list of pairs `(time, value)` meaning that schedule should output
        `value` when `t==time`. All the values for time must be sorted in
        an increasing order. When t is between two times, e.g. `(time_a, value_a)`
        and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
        `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
        time passed between `time_a` and `time_b` for time `t`.
    :param interpolation: (lambda (float, float, float): float)
        a function that takes value to the left and to the right of t according
        to the `endpoints`. Alpha is the fraction of distance from left endpoint to
        right endpoint that t has covered. See linear_interpolation for example.
    :param outside_value: (float)
        if the value is requested outside of all the intervals specified in
        `endpoints` this value is returned. If None then AssertionError is
        raised when outside value is requested.
    """

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints = endpoints

    def value(self, step):
        for (left_t, left), (right_t, right) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if left_t <= step < right_t:
                alpha = float(step - left_t) / (right_t - left_t)
                return self._interpolation(left, right, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value


if __name__ == '__main__':

    exploration1 = PiecewiseSchedule([
        (0, 1.0),
        (1e6, 0.1),
        (2e6, 0.1),
        (5e6, 0.01),
        (8e6, 0.005)
    ], outside_value=0.005)

    exploration2 = PiecewiseSchedule([
        (0, 1.0),
        (1e6, 0.1),
        (2e6, 0.01),
        (8e6, 0.001)
    ], outside_value=0.001)

    steps = np.arange(0, 1e7, 100).astype(int).tolist()
    list1 = list(map(exploration1.value, steps))
    list2 = list(map(exploration2.value, steps))

    df = pd.DataFrame(
        {"steps": steps,
         "e1_0.005": list1,
         "e2_0.001": list2
         }
    )
    df.plot(x='steps', y='e1_0.005', color='r')
    df.plot(x='steps', y='e2_0.001', color='g')
    plt.show()

    # df.to_excel("epsilon_data.xlsx", index=False)






