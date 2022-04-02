import os

import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib import rcParams, rcParamsDefault


def func_inner_product_adcd_x(X, lambda_min):
    # len(X.shape) >= 2 for figures only, not used for grad or Hessian
    x = X[:, :X.shape[1] // 2]
    y = X[:, X.shape[1] // 2:]
    res = np.sum((x * y), axis=1) + np.sum(0.5 * lambda_min * (X * X), axis=1)
    return res


def h_func_inner_product_adcd_x(X):
    # h = g - f
    res = func_inner_product_adcd_x(X, 1) - func_inner_product_adcd_x(X, 0)
    return res


def func_inner_product_adcd_e(X, H_minus):
    # len(X.shape) >= 2 for figures only, not used for grad or Hessian
    x = X[:, :X.shape[1] // 2]
    y = X[:, X.shape[1] // 2:]
    res = np.sum((x * y), axis=1) + np.sum((-0.5 * X @ H_minus) * X, axis=1)
    return res


def h_func_inner_product_adcd_e(X):
    Q = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    Lambda_minus = np.array([[-1, 0], [0, 0]])
    H_minus = Q @ Lambda_minus @ Q
    # h = g - f
    res = func_inner_product_adcd_e(X, H_minus) - func_inner_product_adcd_e(X, np.zeros_like(H_minus))
    return res


def prep_domain_grid():
    X_domain = np.arange(-1, 1.02, 0.1)
    Y_domain = np.arange(-1, 1.02, 0.1)
    X, Y = np.meshgrid(X_domain, Y_domain)
    domain_grid_as_vector = np.concatenate((X.reshape(-1, 1), Y.reshape(-1, 1)), axis=1)
    return X, Y, domain_grid_as_vector


def draw_constraints_bounds(func_to_monitor, convexization_element, img_name):
    fig = plt.figure(figsize=[11.8, 7.8])
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel='$x$', ylabel='$y$')
    ax.azim = -35  # default -60
    ax.elev = 3  # default 30
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    # Plot the surface of the monitored function in the domain
    Z = func_to_monitor(domain_grid_as_vector, convexization_element)
    Z = Z.reshape(X.shape)
    ax.plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    plt.savefig(img_name)
    #plt.show()
    plt.close(fig)


def draw_constraints_bounds_adcd_e_x(func_to_monitor_adcd_e, H_minus, func_to_monitor_adcd_x, lambda_min, img_name):
    fig = plt.figure(figsize=[13, 5.6])
    axs = []
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    axs.append(ax)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    axs.append(ax)
    for ax in axs:
        ax.set(xlabel='$x$', ylabel='$y$')
        ax.azim = -35  # default -60
        ax.elev = 3  # default 30
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    # Get f
    f = func_to_monitor_adcd_x(domain_grid_as_vector, 0)
    f = f.reshape(X.shape)

    # Plot the surface of the monitored function in the domain
    Z = func_to_monitor_adcd_e(domain_grid_as_vector, H_minus)
    Z = Z.reshape(X.shape)
    axs[0].plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)
    # Plot f
    axs[0].plot_surface(X, Y, f, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.05)

    Z = func_to_monitor_adcd_x(domain_grid_as_vector, lambda_min)
    Z = Z.reshape(X.shape)
    axs[1].plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)
    # Plot f
    axs[1].plot_surface(X, Y, f, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.05)

    plt.tight_layout()
    #plt.subplots_adjust(top=1.5, bottom=-0.5, left=0.1, right=0.9)

    plt.savefig(img_name)
    #plt.show()
    plt.close(fig)


def draw_h_func(h_func_inner_product, img_name):
    fig = plt.figure(figsize=[11.8, 7.8])
    ax = fig.add_subplot(projection='3d')
    ax.set(xlabel='$x$', ylabel='$y$')
    ax.azim = -35  # default -60
    ax.elev = 3  # default 30
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    # Plot the surface of the monitored function in the domain
    Z = h_func_inner_product(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    ax.plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    plt.savefig(img_name)
    #plt.show()
    plt.close(fig)


def draw_f_equals_g_minus_h_adcd_x(img_name):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 10})
    rcParams.update({'font.size': 16})
    rcParams['axes.titlesize'] = 30

    fig = plt.figure(figsize=[16.7, 5.5])
    axs = []
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    axs.append(ax)
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    axs.append(ax)
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    axs.append(ax)
    for ax in axs:
        ax.set(xlabel='$x$', ylabel='$y$')
        ax.azim = -35  # default -60
        ax.elev = 3  # default 30
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zlim([-0.8, 1.2])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    Z = func_inner_product_adcd_x(domain_grid_as_vector, 0)
    Z = Z.reshape(X.shape)
    axs[0].plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    Z = func_inner_product_adcd_x(domain_grid_as_vector, 1)
    Z = Z.reshape(X.shape)
    axs[1].plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    Z = h_func_inner_product_adcd_x(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    axs[2].plot_surface(X, Y, Z, color="green", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    plt.savefig(img_name)
    plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def draw_f_equals_g_minus_h_adcd_e(img_name):
    rcParams['pdf.fonttype'] = 42
    rcParams['ps.fonttype'] = 42
    rcParams.update({'legend.fontsize': 10})
    rcParams.update({'font.size': 16})
    rcParams['axes.titlesize'] = 30

    fig = plt.figure(figsize=[16.7, 5.5])
    axs = []
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    axs.append(ax)
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    axs.append(ax)
    ax = fig.add_subplot(1, 3, 3, projection='3d')
    axs.append(ax)
    for ax in axs:
        ax.set(xlabel='$x$', ylabel='$y$')
        ax.azim = -35  # default -60
        ax.elev = 3  # default 30
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zlim([-0.8, 1.2])

    Q = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    Lambda_minus = np.array([[-1, 0], [0, 0]])

    # Prepare domain grid for graph
    X, Y, domain_grid_as_vector = prep_domain_grid()

    H_minus = Q @ (Lambda_minus * 0) @ Q
    Z = func_inner_product_adcd_e(domain_grid_as_vector, H_minus)
    Z = Z.reshape(X.shape)
    axs[0].plot_surface(X, Y, Z, color="blue", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    H_minus = Q @ (Lambda_minus * 1) @ Q
    Z = func_inner_product_adcd_e(domain_grid_as_vector, H_minus)
    Z = Z.reshape(X.shape)
    axs[1].plot_surface(X, Y, Z, color="red", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    Z = h_func_inner_product_adcd_e(domain_grid_as_vector)
    Z = Z.reshape(X.shape)
    axs[2].plot_surface(X, Y, Z, color="green", linewidth=0, antialiased=False, label='f(x)=$g(x)-h(x)$', alpha=0.2)

    plt.savefig(img_name)
    plt.show()
    plt.close(fig)
    rcParams.update(rcParamsDefault)


def build_gif_from_images(images, gif_name):
    with imageio.get_writer(gif_name, mode='I') as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)


if __name__ == "__main__":
    plt.close("all")
    folder = "inner_prod_convexization_gif/"
    try:
        os.mkdir(folder)
    except:
        pass

    images = []
    # 0 is the original inner product func and 1 is the convexed func with ADCD-X
    for idx, i in enumerate(np.linspace(0, 1, num=20)):
        lambda_min = i
        img_name = folder + "adcd_x_" + str(idx) + ".png"
        draw_constraints_bounds(func_inner_product_adcd_x, lambda_min, img_name)
        images.append(img_name)
    # Append the last image multiple times to add delay between loops
    for i in range(15):
        images.append(img_name)

    build_gif_from_images(images, folder + 'adcd_x_convexization.gif')

    images = []
    Q = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    Lambda_minus = np.array([[-1, 0], [0, 0]])
    # 0 is the original inner product func and 1 is the convexed func with ADCD-E
    for idx, i in enumerate(np.linspace(0, 1, num=20)):
        H_minus = Q @ (Lambda_minus * i) @ Q
        img_name = folder + "adcd_e_" + str(idx) + ".png"
        draw_constraints_bounds(func_inner_product_adcd_e, H_minus, img_name)
        images.append(img_name)
    # Append the last image multiple times to add delay between loops
    for i in range(15):
        images.append(img_name)

    build_gif_from_images(images, folder + 'adcd_e_convexization.gif')

    # Combined gif: left image is ADCD-X and right image is ADCD-E
    images = []
    Q = np.array([[-1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), 1 / np.sqrt(2)]])
    Lambda_minus = np.array([[-1, 0], [0, 0]])
    for idx, i in enumerate(np.linspace(0, 1, num=40)):
        lambda_min = i
        H_minus = Q @ (Lambda_minus * i) @ Q
        img_name = folder + "adcd_e_x_" + str(idx) + ".png"
        draw_constraints_bounds_adcd_e_x(func_inner_product_adcd_e, H_minus, func_inner_product_adcd_x, lambda_min, img_name)
        images.append(img_name)
    # Append the last image multiple times to add delay between loops
    for i in range(25):
        images.append(img_name)

    build_gif_from_images(images, folder + 'adcd_e_x_convexization.gif')


    draw_h_func(h_func_inner_product_adcd_x, folder + 'h_func_adcd_x.png')
    draw_h_func(h_func_inner_product_adcd_e, folder + 'h_func_adcd_e.png')

    draw_f_equals_g_minus_h_adcd_x(folder + 'f_equals_g_minus_h_adcd_x.png')
    draw_f_equals_g_minus_h_adcd_e(folder + 'f_equals_g_minus_h_adcd_e.png')

