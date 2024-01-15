import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# set seed
np.random.seed(0)

# original function
def y(x):
    return np.sin(x * np.cos(x)) ** 2

def cubic_spline_interpolate_SCIPY(x, nodes, y, deriv):
    cs = CubicSpline(nodes, y, bc_type="natural")
    return cs(x, nu=deriv)


# number of nodes
num_nodes_list = [5, 10, 30]

# x-axis range
x = np.linspace(-5, 5, 1000)



for mode in ["random", "equidistant"]:
    # subplots
    fig, axs = plt.subplots(3, 1, figsize=(9, 9), sharex=True)

    for i, num_nodes in enumerate(num_nodes_list):
        # nodes
        nodes = np.linspace(-3, 3, num_nodes) if mode == "equidistant" else np.sort(np.random.uniform(-3, 3, size=num_nodes))

        # plot
        axs[i].plot(x, y(x), label="original", color="red")
        axs[i].plot(nodes, y(nodes), "o", color="green")
        axs[i].plot(
            x,
            cubic_spline_interpolate_SCIPY(x, nodes, y(nodes), deriv=0),
            label="interpolated (deriv = 0)",
            color="green",
        )
        axs[i].plot(
            x,
            cubic_spline_interpolate_SCIPY(x, nodes, y(nodes), deriv=1),
            "--",
            label="interpolated (deriv = 1)",
            alpha=0.8,
            linewidth=0.5,
        )
        axs[i].plot(
            x,
            cubic_spline_interpolate_SCIPY(x, nodes, y(nodes), deriv=2),
            "--",
            label="interpolated (deriv = 2)",
            alpha=0.8,
            linewidth=0.5,
        )
        axs[i].set_xlim(-4, 4)
        axs[i].set_ylim(-2, 2)
        # axs[i].set_title(f'Number of nodes = {num_nodes}')

    axs[2].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4)

    # Save the plot
    fig.savefig(f"cubic_spline_scipy_{mode}.png", dpi=100, bbox_inches="tight")
