import matplotlib.pylab as plt
import numba
import numpy as np
import seaborn as sns
from pyDOE import *
import statistics
import time
sns.set()


# Image size (pixels)
# HEIGHT, WIDTH = 400, 600
HEIGHT, WIDTH = 6, 6

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1
hits = 0
samples = 0
methods = ["pure", "pure antithetic", "LHS", "LHS antithetic", "ortho", "ortho antithetic"]


@numba.jit(nopython=True)
def mandelbrot(c, max_iterations):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z*z + c
        n += 1
    return n


def get_area(total_colors, darts, antithetic):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Calculate area with that number.
    """
    hits_i = 0
    hits_a = 0

    random_nrs_i = np.random.uniform(0, 1, darts)
    random_nrs_a = 1-random_nrs_i

    # throw darts
    for i in range(len(random_nrs_i)):

        x = int(round(random_nrs_i[i] * len(total_colors)))
        x_a = int(round(random_nrs_a[i] * len(total_colors)))

        if x > len(total_colors) - 1:
            x = int(x - 1)
        if x_a > len(total_colors) - 1:
            x_a = int(x - 1)

        # check if it landed in the mandelbrot
        color_i = total_colors[x]
        color_a = total_colors[x_a]

        if color_i == 0:
            hits_i += 1
        if color_a == 0:
            hits_a += 1

    area_i = (hits_i / darts) * 6
    area_a = (hits_a / darts) * 6

    if antithetic is True:
        return (area_i + area_a) / 2
    return area_i


def get_area_lhs(total_colors, darts, ortho, antithetic):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Get random numbers through the LHS method.
    """
    hits = 0
    hits_a = 0
    HEIGHT, WIDTH = 6, 6

    # Get list of latin-hybercube sampled values in 2 dimensions (X and Y)
    lhd = lhs(2, samples=darts)

    if ortho is False:
        total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)
    else:
        HEIGHT = HEIGHT / 2
        WIDTH = WIDTH / 2
        total_colors = np.array(total_colors).reshape(int(HEIGHT), int(WIDTH))

    hits, hits_a = calculatehits_coordinates(lhd[:, 0], lhd[:, 1], total_colors)

    area = (hits / darts) * 6
    area_a = (hits_a / darts) * 6
    if ortho is False and antithetic is True:
        return (area + area_a) / 2
    elif ortho is False and antithetic is False:
        return area
    elif ortho is True:
        return hits


def generate_o(total_colors, major, antithetic):
    values_i = []
    values_r = []

    min_i, min_r = 0, 0
    max_i, max_r = 1, 1
    # major = np.sqrt(major)

    samples = major * major
    darts = samples

    scale_i = (max_i - min_i) / samples
    scale_r = (max_r - min_r) / samples
    total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)

    xlist = [[0 for i in range(major)] for j in range(major)]
    ylist = [[0 for i in range(major)] for j in range(major)]

    m = 0
    hits = 0
    hits_a = 0

    for i in range(major):
        for j in range(major):
            xlist[i][j] = ylist[i][j] = m
            m += 1

    np.random.shuffle(xlist)
    np.random.shuffle(ylist)


    for i in range(major):
        for j in range(major):
            values_i.append(min_i + scale_i * (xlist[i][j] + np.random.random() ))
            values_r.append(min_r + scale_r * (ylist[j][i] + np.random.random() ))

    hits, hits_a = calculatehits_coordinates(values_i, values_r, total_colors)

    area = (hits / darts) * 6
    area_a = (hits_a / darts) * 6

    if antithetic is True:
        return (area + area_a) / 2
    return area


def calculatehits_coordinates(values_i, values_r, total_colors):
    hits = 0
    hits_a = 0

    for j in range(len(values_i)):

        # Get x and y coordinate
        x, y = values_i[j], values_r[j]
        x_a = 1 - x
        y_a = 1 - y
        x, y = (int(round(x * WIDTH))), (int(round(y * HEIGHT)))
        x_a, y_a = (int(round(x_a * WIDTH))), (int(round(y_a * HEIGHT)))
        if x > WIDTH - 1:
            x = int(WIDTH - 1)
        if y > HEIGHT - 1:
            y = int(HEIGHT - 1)
        if x_a > WIDTH - 1:
            x_a = int(WIDTH - 1)
        if y_a > HEIGHT - 1:
            y_a = int(HEIGHT - 1)
        color = total_colors[y][x]
        color_a = total_colors[y_a][x_a]
        if color == 0:
            hits += 1
        if color_a == 0:
            hits_a += 1

    return hits, hits_a


def compare_area(iterations, darts):
    """
    Compare estimated area for different sample sizes and iterations.
    """
    area_is = get_area(make_mandelbrot(iterations), darts, antithetic=False)

    compare_list = []
    j_list = []
    area_js_list = []
    for j in range(iterations - 1):
        iterations = j + 1
        area_js = get_area(make_mandelbrot(j + 1), darts)
        difference = area_js - area_is
        compare_list.append(difference)
        j_list.append(j + 1)
        area_js_list.append(area_js)

    plt.xlabel("j")
    plt.ylabel("A_j,s")
    plt.plot(j_list, area_js_list)
    area_line = []
    for i in range(len(j_list)):
        area_line.append(1.507)
    plt.plot(j_list, area_line)
    plt.show()

    print(area_js_list)
    print(sum(area_js_list)/len(area_js_list))

    return compare_list


def compare_i(max_iterations, darts):

    area_list = []
    for i in range(max_iterations - 1):
        area_i = get_area(make_mandelbrot(i + 1), darts)[0]
        area_list.append(area_i)

    return area_list


def all_methods(iterations, max_darts, method):

    area_list = []
    n = 1

    if method == "pure":
        for s in range(n):
            area_i = get_area(make_mandelbrot(iterations), max_darts, antithetic=False)
            area_list.append(area_i)
    elif method == "pure antithetic":
        for s in range(n):
            area_i = get_area(make_mandelbrot(iterations), max_darts, antithetic=True)
            area_list.append(area_i)
    elif method == "LHS":
        for s in range(n):
            area_i = get_area_lhs(make_mandelbrot(iterations), max_darts, ortho=False, antithetic=False)
            area_list.append(area_i)
    elif method == "LHS antithetic":
        for s in range(n):
            area_i = get_area_lhs(make_mandelbrot(iterations), max_darts, ortho=False, antithetic=True)
            area_list.append(area_i)
    elif method == "ortho":
        for s in range(n):
            area_i = generate_o(make_mandelbrot(iterations), int(np.sqrt(max_darts)), antithetic=False)
            area_list.append(area_i)
    elif method == "ortho antithetic":
        for s in range(n):
            area_i = generate_o(make_mandelbrot(iterations), int(np.sqrt(max_darts)), antithetic=True)
            area_list.append(area_i)

    return area_list


def make_mandelbrot(iterations):
    """
    Return a list of colors in the Mandelbrot set.
    """

    total_colors = []
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            # Number of iterations
            m = mandelbrot(c, iterations)

            # Get color of these iterations
            color = 255 - int(m * 255 / iterations)
            total_colors.append(color)

    return total_colors


def calculate_variance(n, darts, method):
    """
    Calculate variance of all three methods.
    """

    vars, means, stds = [], [], []
    areas = []
    d = 0.05
    i = 1

    import time
    tic = time.clock()
    samplevariance = 1

    while samplevariance > d or i < 100:

        areas.append(all_methods(n, darts, method)[0])

        if i > 2:
            vars.append(np.var(areas))
            stds.append(np.std(areas))
            means.append(np.mean(areas))
            samplevariance = (1.96 * statistics.variance(areas) / np.sqrt(i))
            print(method,  "mean", round(np.mean(areas), 3))
        i += 1

    toc = time.clock()

    return vars, means, i, toc - tic, stds


def make_barplot_var(vars, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.27
    ticks = 6
    ind = np.arange(ticks)

    yvals = vars
    rects1 = ax.bar(ind, yvals, width, color='r')

    ax.set_ylabel('Scores')
    ax.set_xticks(ind)
    ax.set_xticklabels(["Pure", "Pure antithetic",  "LHS", "LHS antithetic", "orthogonal", "orthogonal antithetic"])

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.00*h, round(h, 3),
                    ha='center', va='bottom')

    autolabel(rects1)
    plt.title(title)
    plt.show()

def compare_area(iterations, darts):
    area_is = get_area(make_mandelbrot(iterations), darts)[0]

    compare_list = []
    j_list = []
    area_js_list = []
    for j in range(iterations - 1):
        iterations = j + 1
        area_js = get_area(make_mandelbrot(j + 1), darts)[0]
        difference = abs(area_js - area_is)
        compare_list.append(difference)
        j_list.append(j + 1)
        area_js_list.append(area_js)

    return compare_list


def compare_i(max_iterations, darts):

    area_list = []
    for i in range(max_iterations - 1):
        area_i = get_area(make_mandelbrot(i + 1), darts)[0]
        area_list.append(area_i)

    return area_list


def compare_i_variance(n, max_iterations, darts):
    """
    Compute the variance n times and plot the mean, max and min value of the variances
    """

    areas = []
    for i in range(n):
        area_list = compare_i(max_iterations, samples)
        areas.append(area_list)

    variances = []
    for i in range(max_iterations - 1):
        variance_list = []
        for j in range(n):
            variance_list.append(areas[j][i])
        variances.append(np.var(variance_list))


    plt.xlabel("Number of iterations")
    plt.ylabel("Variance")
    plt.plot(range(1, max_iterations), variances)
    plt.show()


def compare_s(iterations, max_darts):

    area_list = []
    for s in range(max_darts - 1):
        area_i = get_area(make_mandelbrot(iterations), s + 1)[0]
        area_list.append(area_i)

    return area_list


def compare_s_variance(n, iterations, max_darts):
    """
    Compute the variance n times and plot the mean, max and min value of the variances
    """

    areas = []
    for i in range(n):
        area_list = compare_s(iterations, max_darts)
        areas.append(area_list)

    variances = []
    for i in range(max_darts - 1):
        variance_list = []
        for j in range(n):
            variance_list.append(areas[j][i])
        variances.append(np.var(variance_list))

    plt.xlabel("Number of samples")
    plt.ylabel("Variance")
    plt.plot(range(1, max_darts), variances)
    plt.show()


def compare_methods():
    iterations = 30
    darts = 289

    totalvars, totalmeans, iteration, CLT_iterations_list, times, stds = [], [], [], [], [], []
    for method in methods:

        for i in range(2):
            vars, means, CLT_iterations, time, std = calculate_variance(iterations, darts, method)
        totalvars.append(np.mean(vars))
        totalmeans.append(np.mean(means))
        times.append(np.mean(time))
        stds.append(np.mean(std))
        CLT_iterations_list.append(np.mean(CLT_iterations))
    print(totalvars, "vars")
    print(totalmeans, "means")
    print(times, "times")
    print(stds, "stds")
    print(CLT_iterations_list)

    make_barplot_var(vars, "Variance")
    make_barplot_var(totalmeans, "Mean estimated area")


if __name__ == '__main__':

    compare_methods()
