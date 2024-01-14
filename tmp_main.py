import numpy as np
import matplotlib.pyplot as plt


def natural_cubic_spline_interp(xs :np.ndarray , ys: np.ndarray, eval_xs: np.ndarray) -> np.ndarray:
    """## Interpolate using a natural cubic spline. 

    xs: spline's nodes' x-axis coordinates
    ys: spline's nodes' y-axis coordinates
    eval_xs: x-axis coordinates to evaluate the spline at

    returns: y-axis coordinates of the spline evaluated at eval_xs

    ## --------------------- Algorithm ---------------------
    #### (based on "Przegląd metod i algorytmów numerycznych, Cz. 1" by Janina i Michał Jankowscy) 

    We have N intervals, and 4 coefficients for each interval:
        a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3 
    for each interval [x_i, x_i+1]
    for (i = 0, 1, ..., N-1)

    We are looking for the coefficients a, b, c and d for each interval of the spline:
        a_i is the y-axis coordinate of the spline at x_i (we already know this)
        b_i = y_i+1 - y_i / h_i - h_i / 3 * (c_i+1 + 2*c_i) 
        c_i = ??? (matrix equation unknowns)
        d_i = (c_i+1 - c_i) / (3*h_i)
    for (i = 0, 1, ..., N-1)
    we first calculate the c_i coefficients, then the b_i and d_i coefficients

    (helper notation) h_i = x_i+1 - x_i (we already know this)
    for (i = 0, 1, ..., N-1)

    mat is a tridiagonal matrix with 2 on the main diagonal and w_i and u_i on the sub and super diagonal:
        w_i = h_i / (h_i-1 + h_i)
    for (i = 1, 2, ..., N-1) 
        u_i = h_i-1 / (h_i-1 + h_i)
    for (i = 2, 3, ..., N)    
    
    v is the right hand side of the matrix equation
        v_i = ((y_i+1 - y_  i) / h_i - (y_i - y_i-1) / h_i-1) / (h_i + h_i-1)
    for (i = 1, 2, ..., N)
    
    The matrix equation is:
        mat @ (c/3) = v  
            or 
        c = inv(mat) @ v * 3  
    """

    ######################################################################################
    ##################### calculate the coefficients of the spline ######################
    ######################################################################################
    
    n_nodes = len(xs) # N+1 nodes
    n_intervals = n_nodes - 1 # N intervals

    h = np.full(n_nodes, np.nan)
    for i in range(n_intervals):
        h[i] = xs[i+1] - xs[i]

    w = np.full(n_nodes, np.nan)
    for i in range(1, n_intervals-2 + 1): # 1 to N-2 (inclusive)
        w[i] = h[i] / (h[i-1] + h[i])
        
    u = np.full(n_nodes, np.nan)
    for i in range(2, n_intervals-1 + 1): # 2 to N-1 (inclusive)
        u[i] = h[i-1] / (h[i-1] + h[i])

    # calculate the tri-diagonal matrix (N-1 x N-1)
    mat = (
        np.diag(np.full(n_intervals-1, 2)) # 2's on the main diagonal 
        + np.diag(w[1: n_intervals-2 + 1], 1) # w_i's on the super diagonal
        + np.diag(u[2: n_intervals-1 + 1], -1) # u_i's on the sub diagonal
        )

    # calculate the right hand side of the equation
    v = np.full(n_nodes, np.nan)
    for i in range(1, n_intervals-1 + 1): # 1 to N-1 (inclusive)
        v[i] = ((ys[i+1] - ys[i]) / h[i] - (ys[i] - ys[i-1]) / h[i-1]) / (h[i] + h[i-1])

    # calculate the c_i coefficients
    # c = np.linalg.inv(mat) @ out * 3
    c = np.linalg.inv(mat) @ v[1: n_intervals-1 + 1] * 3

    # set c_0 and c_N to 0 (a property of natural splines)
    c = np.insert(c, 0, 0)
    c = np.append(c, 0)

    # calculate the b_i and d_i coefficients
    b = np.full(n_nodes, np.nan)
    d = np.full(n_nodes, np.nan)
    for i in range(n_intervals):
        b[i] = (ys[i+1] - ys[i]) / h[i] - h[i] / 3 * (c[i+1] + 2*c[i])
        d[i] = (c[i+1] - c[i]) / (3*h[i])

    # calculate the a_i coefficients
    a = ys

    ######################################################################################
    ######### calculate the y-axis coordinates of the spline evaluated at eval_xs ###########
    ######################################################################################

    # find the interval that each eval_xs point belongs to
    # (the interval that each eval_xs point belongs to is the interval that contains it)
    eval_xs = np.sort(eval_xs)
    interval_indices = np.searchsorted(xs, eval_xs, side='right') - 1
    interval_indices = np.clip(interval_indices, 0, n_intervals-1) # clip to [0, N-1] (prevent out of bounds due to numerical errors)
    assert np.all(interval_indices >= 0) and np.all(interval_indices < n_intervals), "ERR: interval_indices out of bounds"

    # calculate the y-axis coordinates of the spline evaluated at eval_xs
    eval_ys = np.zeros(len(eval_xs))

    for i in range(len(eval_xs)):
        # find the interval that the eval point belongs to
        interval_index = interval_indices[i]

        # calculate the y-axis coordinate of the spline evaluated at eval_xs
        eval_ys[i] = (
            a[interval_index] 
            + b[interval_index] * (eval_xs[i] - xs[interval_index]) 
            + c[interval_index] * (eval_xs[i] - xs[interval_index])**2 
            + d[interval_index] * (eval_xs[i] - xs[interval_index])**3
            )
        
    return eval_ys


def plot_spline(xs :np.ndarray , ys: np.ndarray, eval_xs: np.ndarray, eval_ys: np.ndarray, ground_truth: np.ndarray):
    """Plot the spline and its nodes"""

    plt.plot(xs, ys, 'o', label='spline nodes')
    plt.plot(eval_xs, eval_ys, label='spline')
    plt.plot(eval_xs, ground_truth, label='ground truth')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-2, 2)
    plt.xlim(-4, 4)
    plt.legend()
    plt.show()


gt_func = lambda x: np.sin(x * np.cos(x))**2
xs = np.linspace(-3, 3, 10)
ys = gt_func(xs)
eval_xs = np.linspace(-4, 4, 1000)
eval_ys = natural_cubic_spline_interp(xs, ys, eval_xs)
plot_spline(xs, ys, eval_xs, eval_ys, gt_func(eval_xs))

# TODO check why the ends are different from the scipy
# TODO check  if not-evenly spaced nodes work
