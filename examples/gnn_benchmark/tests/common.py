import numpy as np


def check_equal(expected_pred, pred, name=None):
    print('\n' + name if name else '')
    print('Calculated: \n', pred)
    print('Expected: \n', expected_pred)
    if not np.allclose(pred, expected_pred, atol=1e-6):
        max_err_abs = np.abs(pred - expected_pred).max()
        print("Abs error: ", max_err_abs)
        max_err_rel = max_err_abs / np.abs(expected_pred).max()
        print("Rel error: ",
              max_err_rel)
        assert False, f"{name} abs: {max_err_abs}, rel: {max_err_rel}"
