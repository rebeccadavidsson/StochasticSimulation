from PIL import Image, ImageDraw
import random
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numba
import numpy as np
import seaborn as sns
from pyDOE import *
import scipy.linalg
sns.set()




@numba.jit(nopython=True)
def mandelbrot(c, max_iterations):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z*z + c
        n += 1
    return n


# for a in range(-10, 10, 5):
#     for b in range(-10, 10, 5):
#         c = complex(a / 10, b / 10)
#         # print(c, mandelbrot(c))


# Image size (pixels)
# WIDTH = 600
# HEIGHT = 400
WIDTH = 40
HEIGHT = 40

methods = ["pure", "pure antithetic", "LHS", "LHS antithetic", "ortho", "ortho antithetic"]

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1

hits = 0
samples = 0


def get_area(total_colors, darts):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Calculate area with that number.
    """
    hits = 0

    # throw darts
    for i in range(darts):

        # check if it landed in the mandelbrot
        color = random.choice(total_colors)

        if color == 0:
            hits += 1

    area = (hits / darts) * 6

    return area


def get_area_antithetic(total_colors, darts):
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

    # Unbiased estimator of area
    return (area_i + area_a) / 2

def get_area_lhs(total_colors, darts, ortho):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Get random numbers through the LHS method.
    """
    hits = 0

    # Get list of latin-hybercube sampled values in 2 dimensions (X and Y)
    lhd = lhs(2, samples=darts)

    # TODO, dit is gehardcode!!!
    HEIGHT, WIDTH = 40, 40
    if ortho is False:
        total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)
    else:
        HEIGHT = HEIGHT / 2
        WIDTH = WIDTH / 2
        total_colors = np.array(total_colors).reshape(int(HEIGHT), int(WIDTH))

    count = 0
    for j in range(len(lhd)):

        # Get x and y coordinate
        x, y = lhd[j][0], lhd[j][1]
        x = (int(round(x * WIDTH)))
        y = (int(round(y * HEIGHT)))
        if x > WIDTH - 1:
            x = int(WIDTH - 1)
        if y > HEIGHT - 1:
            y = int(HEIGHT - 1)
        color = total_colors[y][x]

        if color == 0:
            hits += 1

    area = (hits / darts) * 6

    if ortho is False:
        return area
    return hits


def get_area_lhs_a(total_colors, darts, ortho):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Get random numbers through the LHS method.
    """
    hits = 0
    hits_a = 0
    HEIGHT, WIDTH = 40, 40

    # Get list of latin-hybercube sampled values in 2 dimensions (X and Y)
    lhd = lhs(2, samples=darts)

    if ortho is False:
        total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)
    else:
        HEIGHT = HEIGHT / 2
        WIDTH = WIDTH / 2
        total_colors = np.array(total_colors).reshape(int(HEIGHT), int(WIDTH))


    for j in range(len(lhd)):

        # Get x and y coordinate
        x, y = lhd[j][0], lhd[j][1]
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

    area = (hits / darts) * 6
    area_a = (hits_a / darts) * 6
    if ortho is False:
        return (area + area_a) / 2
    return hits


def generate_o(major):
    values_i = []
    values_r = []
    min_r = 0
    max_r = 1
    min_i = 0
    max_i = 1
    diff_i = 1
    diff_r = 1

    samples = major * major

    scale_i = (diff_i) / samples
    scale_r = (diff_r) / samples

    xlist = [[0 for i in range(major)] for j in range(major)]
    ylist = [[0 for i in range(major)] for j in range(major)]

    m = 0

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


    if len(set(values_i)) == len(values_i):
        print("chilllll")

    if len(set(values_r)) == len(values_r):
        print("wiehoeee")

    return values_i, values_r


# @jit
# def ortho_sampling(n=1000):
#     samples = n**2
#     xlist = np.zeros((n,n))
#     ylist = np.zeros((n,n))
#     scale = 4.0 / samples
#     m = 0
#     for i in range(n):
#         for j in range(n):
#             m += 1
#             xlist[i,j] = ylist[i,j] = m
#     for k in range(samples):
#         for i in range(n):
#             xlist[i,:] = np.random.permutation(xlist[i,:])
#             ylist[i,:] = np.random.permutation(ylist[i,:])
#         for i in range(n):
#             for j in range(n):
#                 xlist[i,j] = -2 + scale*(xlist[i,j] + np.uniform(1))
#                 ylist[i,j] = -2 + scale*(ylist[i,j] + np.uniform(1))
#     return xlist, ylist


def get_area_ortho(total_colors, darts):
    """
    Throw darts in the domain and count how many hit the mandelbrot.
    Calculate area with orthogonal sampling.
    """
    hits = 0


    # Split into four matrixes
    x = np.split(np.array(total_colors), 4)

    # get part
    part = int(np.floor(darts / 4))
    hits = 0
    # throw darts
    for i in range(part):

        # Get a sample in every matrix
        for j in x:
            hits += get_area_lhs(j, 1, ortho=True)

    area = (hits / darts) * 6
    # print("AREA", area)
    return area

def get_area_ortho_a(total_colors, darts):
    """
    Throw darts in the domain and count how many hit the mandelbrot.
    Calculate area with orthogonal sampling.
    """
    hits = 0
    hits_a = 0


    # Split into four matrixes
    x = np.split(np.array(total_colors), 4)

    # get part
    part = int(np.floor(darts / 4))
    hits = 0
    # throw darts
    for i in range(part):

        # Get a sample in every matrix
        for j in x:
            hits += get_area_lhs(j, 1, ortho=True)
            hits_a += get_area_lhs_a(j, 1, ortho=True)

    area = (hits / darts) * 6
    area_a = (hits_a / darts) * 6
    # print("AREA", area)
    return (area + area_a) / 2


def compare_area(iterations, darts):
    area_is = get_area(make_mandelbrot(iterations), darts)

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

        # if difference < compare_list[-1]:

    # plt.xlabel("j")
    # plt.ylabel("A_j,s - A_i,s")
    # plt.plot(j_list, compare_list)
    # plt.show()
    #
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
        area_i = get_area(make_mandelbrot(i + 1), darts)
        area_list.append(area_i)

    plt.xlabel("Number of iterations")
    plt.ylabel("Area_i,s")
    plt.plot(range(1, max_iterations), area_list)
    plt.show()


def compare_s(iterations, max_darts, method):

    area_list = []
    n = max_darts

    if method == "pure":
        for s in range(n):
            area_i = get_area(make_mandelbrot(iterations), s + 1)
            area_list.append(area_i)
    elif method == "pure antithetic":
        for s in range(n):
            area_i = get_area_antithetic(make_mandelbrot(iterations), s + 1)
            area_list.append(area_i)
    elif method == "LHS":
        for s in range(n):
            area_i = get_area_lhs(make_mandelbrot(iterations), s + 1, ortho=False)
            area_list.append(area_i)
    elif method == "LHS antithetic":
        for s in range(n):
            area_i = get_area_lhs_a(make_mandelbrot(iterations), s + 1, ortho=False)
            area_list.append(area_i)
    elif method == "ortho":
        for s in range(n):
            area_i = get_area_ortho(make_mandelbrot(iterations), s + 1)
            area_list.append(area_i)
    elif method == "ortho antithetic":
        for s in range(n):
            area_i = get_area_ortho(make_mandelbrot(iterations), s + 1)
            area_list.append(area_i)


    # plt.xlabel("Number of iterations")
    # plt.ylabel("Area_i,s")
    # plt.plot(range(1, max_darts), area_list)
    # area_line = []
    # print(len(area_list))
    # for i in range(len(area_list)):
    #     area_line.append(1.507)
    # plt.plot(range(1, max_darts), area_line)
    # plt.show()
    return area_list


def make_mandelbrot(iterations):
    """
    Return a list of colors in the Mandelbrot set.
    """
    # im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    # draw = ImageDraw.Draw(im)

    total_colors = []
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            # Compute the number of iterations
            m = mandelbrot(c, iterations)

            # The color depends on the number of iterations
            color = 255 - int(m * 255 / iterations)

            # plot the point
            # draw.point([x, y], (color, color, color))
            total_colors.append(color)

    # im.show()

    return total_colors


def make_plot():
    """
    Plot the number of hits against number of iterations.
    """

    total = []
    darts = 10
    for i in range(1, 40):
        total.append(compare_area(i, 10))

    plt.ylabel("Number of hits with " + str(darts) + "darts")
    plt.xlabel("Number of iterations")
    plt.plot(total)
    plt.show()

def make_3dplot(max_iterations, max_darts):
    darts = []
    iterations = []
    area = []
    # darts = np.array([])
    # iterations = np.array([])
    # area = np.array([])
    for i in range(max_iterations - 1):
        for s in range(max_darts - 1):
            area_is = get_area(make_mandelbrot(i + 1), s + 1)
            iterations.append(i + 1)
            darts.append(s + 1)
            area.append(area_is)
            # np.append(iterations, i + 1)
            # np.append(darts, s + 1)
            # np.append(area, area_is)

    # scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(iterations, darts, area)
    print(iterations)
    print(darts)
    print(area)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Darts')
    ax.set_zlabel('Area')

    plt.show()

    # # surface plot <--- GAAT NOG NIET GOED!!
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    #
    # iterations = []
    # darts = []
    # area = []
    # for i in range(max_iterations - 1):
    #     iterations.append(i + 1)
    #     area_s = []
    #     for s in range(max_darts - 1):
    #         area_is = get_area(make_mandelbrot(i + 1), s + 1)
    #         area_s.append(area_is)
    #     area.append(area_s)
    # for s in range(max_darts - 1):
    #     darts.append(s + 1)
    #
    # print(iterations)
    # print(darts)
    # print(area)
    #
    # # Plot the surface.
    # surf = ax.plot_surface(np.asarray(iterations), np.asarray(darts), np.asarray(area), cmap=cm.coolwarm)
    #
    # # Add a color bar which maps values to colors.
    # fig.colorbar(surf, shrink=0.5, aspect=5)
    #
    # plt.show()


def calculate_variance(n, darts, method):
    """
    Calculate variance of all three methods.
    """

    # n = ((1.96/0.05)^2) * ((area/1.506)-1)


    vars, means = [], []
    areas = [0,1]
    d = 0.05
    i = 1

    while (1.96 * np.var(areas) / np.sqrt(i)) > d:
        # total = []
        # for i in range(min, max):
        #     total.append(compare_s(i, 10))
        areas = compare_s(n, darts, method)
        vars.append(np.var(areas))
        means.append(np.mean(areas))
        print(method, "var", round(np.var(areas), 2), "mean", round(np.mean(areas), 2))

        print("formula", (1.96 * np.var(areas) / np.sqrt(i)))
        i += 1

    print("_________________________")
    return vars, means, i


def make_barplot(vars, means):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.27
    ticks = 6
    ind = np.arange(ticks)

    yvals = vars
    rects1 = ax.bar(ind, yvals, width, color='r')
    # zvals = means
    # rects2 = ax.bar(ind+width, zvals, width, color='g')

    ax.set_ylabel('Scores')
    ax.set_xticks(ind)
    ax.set_xticklabels(["Pure random", "pure antithetic",  "LHS", "LHS antithetic", "orthogonal", "orthogonal antithetic"])
    # ax.legend((rects1[0], rects2[0]), ('Variance', 'Mean estimated area'), loc="upper center", bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.00*h, round(h,3),
                    ha='center', va='bottom')

    autolabel(rects1)
    # autolabel(rects2)
    plt.title("iterations = 20, darts = 30")

    plt.show()


def make_linegraph():

    iterations = 5
    darts = 5
    totalvars, totalmeans, iteration, CLT_iterations_list = [], [], [], []
    for method in methods:
        for i in range(1, 3):
            vars, means, CLT_iterations = calculate_variance(iterations, darts, method)
            totalvars.append(vars)
            totalmeans.append(means)
            iteration.append(i)
            CLT_iterations_list.append(CLT_iterations)
            iterations += 5
            darts += 5

    plt.plot(iteration, totalmeans)
    plt.legend(methods)
    plt.xlabel("Iterations")
    plt.ylabel("Mean estimated area")
    plt.show()

    plt.plot(iteration, totalvars)
    plt.legend(methods)
    plt.xlabel("Iterations")
    plt.ylabel("Variance")
    plt.show()


def experiment_a(n, darts, maxiter, sampling):

    hit_area = []
    for i in range(n):

        rand_values = generate_a(darts, sampling)

        hit, total = iterate (maxiter, rand_values[0], rand_values[1])
        hit_a, total_a = iterate (maxiter, rand_values[2], rand_values[3])

        area = hit / total * total_volume
        area_a = hit_a / total_a * total_volume

        hit_area.append((area + area_a) / 2)

        if i % 50 == 0:
            print(i)

    print (hit + hit_a, total + total_a)

    meanx = mean(hit_area)
    varx = var(hit_area)
    stdx = std(hit_area)

    return meanx, varx, stdx


# antithetic VARIABLES!!!!!!
def generate_a(darts, sampling):
    values_i_a = []
    values_r_a = []
    norm_i = []
    norm_r = []
    norm_i_a = []
    norm_r_a = []

#     Generate variables
    if sampling == "lhs":
        values_i, values_r = generate_lhs(darts)
    elif sampling == "o":
        values_i, values_r = generate_o(darts)
        darts = darts * darts
    else:
        values_i, values_r = generate_r(darts)

#     Normalize random variables
    for i in range(darts):
        norm_i.append((values_i[i] - min_i) / (max_i - min_i))
        norm_r.append((values_r[i] - min_r) / (max_r - min_r))

#     Generate normalized antithetic variables
    for i in range(darts):
        norm_i_a.append(1 - norm_i[i])
        norm_r_a.append(1 - norm_r[i])

#     Denormalize antithetic variables
    for i in range(darts):
        values_i_a.append(norm_i_a[i] * (max_i - min_i) + min_i)
        values_r_a.append(norm_r_a[i] * (max_r - min_r) + min_r)

    return values_i, values_r, values_i_a, values_r_a


if __name__ == '__main__':

    generate_o(17)

# all rebecca's dingen even gecomment :)
    # # # Barplot :)
    # iterations = 20
    # darts = 30
    # #
    # # calculate_variance(iterations, darts, "ortho")
    # totalvars, totalmeans, CLT_iterations_list = [], [], []
    # for method in methods:
    #     for i in range(8):
    #         vars, means, CLT_iterations = calculate_variance(iterations, darts, method)
    #     totalvars.append(np.mean(vars))
    #     totalmeans.append(np.mean(means))
    #     CLT_iterations_list.append(np.mean(CLT_iterations))
    #     # print(i, CLT_iterations)
    # print(totalvars)
    # print(totalmeans)
    # print(CLT_iterations_list)
    #
    # # make_barplot(totalvars, totalmeans)
    # make_barplot(totalvars, CLT_iterations_list)
    #
    # plt.bar([1,2,3,4, 5, 6], CLT_iterations_list)
    # plt.ylabel("Samples")
    # # plt.xlabel(["Pure random", "LHS", "orthogonal"]) fig = plt.figure() ax =
    # # fig.add_subplot(111) ticks = 3 ind = np.arange(ticks)
    # # ax.set_ylabel('Scores') ax.set_xticks(ind) ax.set_xticklabels(["Pure
    # # random", "LHS", "orthogonal"]) ax.legend((rects1[0], rects2[0]),
    # # ('Variance', 'Mean estimated area'), loc="upper center",
    # # bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)
    # plt.show()
    #
    # # make_linegraph()
    #
    #
    # # iterations = 100
    # # darts = 50
    # # compare_area(iterations, darts)
    # # areas = []
    # # for i in range(10):
    # #     areas.append(compare_s(iterations, darts, "pure"))
    # #
    # # plt.plot(areas)
    # # plt.show()
