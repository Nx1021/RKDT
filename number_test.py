import numpy as np

ratio = np.array(
    [[15.81,69.14],
    [63.3,86.09],
    [16.68,65.12],
    [25.24,61.44],
    [65.65,73.06],
    [50.17,8.43],
    [49.62,55.37],
    [39.67,69.84]]) / 100


# ratio = np.array(
# [[99.23, 43.62],
# [99.81, 99.90],
# [99.21, 86.86],
# [99.90, 95.47],
# [99.30, 79.34],
# [96.92, 96.43],
# [98.02, 52.58],
# [99.34, 99.15],
# [98.45, 95.66],
# [100.0, 81.92],
# [99.18, 98.88],
# [98.27, 99.33],
# [99.42, 92.41]]) / 100


for ratios in ratio:
    number = np.arange(100, 1214, dtype=np.int32)
    mul = np.matmul(number[..., np.newaxis], ratios[np.newaxis, ...])

    delta = np.sum(np.abs(mul - np.round(mul)), axis=1) / number
    idx = np.argmin(delta)
    print(number[idx], np.round(mul[idx]))
