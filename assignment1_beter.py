from PIL import Image, ImageDraw
import random
import matplotlib.pylab as plt
import numba
import numpy as np
import seaborn as sns
from pyDOE import *
from random import randrange
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
WIDTH = 600
HEIGHT = 400

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
    hit_list = []

    # throw darts
    for i in range(darts):

        # check if it landed in the mandelbrot
        # color = random.choice(total_colors)

        # check if it landed in the mandelbrot + remember where darts where thrown
        index = randrange(len(total_colors))
        color = total_colors[index]
        hit_list.append(index)

        if color == 0:
            hits += 1

    area = (hits / darts) * 6

    return area, hit_list


def get_area_lhs(total_colors, darts):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Get random numbers through the LHS method.
    """
    hits = 0
    hit_list = []

    # Get list of latin-hybercube sampled values in 2 dimensions (X and Y)
    lhd = lhs(2, samples=darts)

    total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)
    color_row = len(total_colors[0])

    count = 0
    for j in range(len(lhd)):

        # Get x and y coordinate
        x, y = lhd[j][0], lhd[j][1]
        x = (int(round(x * WIDTH)))
        y = (int(round(y * HEIGHT)))
        if x > WIDTH - 1:
            x = WIDTH - 1
        if y > HEIGHT - 1:
            y = HEIGHT - 1
        color = total_colors[y][x]

        if color == 0:
            hits += 1

        hit_list.append((y) * color_row + x)

    area = (hits / darts) * 6

    return area, hit_list


def generate_o(major):
    values_i = []
    values_r = []
    max_i = 10
    min_i = 1
    max_r = 2
    min_r = 5

    samples = major * major

    scale_i = (max_i - min_i) / samples
    scale_r = (max_r - min_r) / samples

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
            values_r.append(min_r + scale_r * (xlist[j][i] + np.random.random() ))

    return values_i, values_r


def get_area_ortho(total_colors, darts, n):
    """
    Throw darts in the domain and count how many hit the mandelbrot.
    Calculate area with orthogonal sampling.
    """
    hits = 0
    hit_list = []

    # Split into n*n matrices, n square so you get a nice grid
    x = np.split(np.array(total_colors), n * n)

    # get part
    part = int(np.floor(darts / 4))

    # throw darts
    for i in range(part):

        # Get a sample in every matrix
        for j in x:
            # # check if it landed in the mandelbrot
            # color = random.choice(total_colors)

            # check if it landed in the mandelbrot + remember where darts where thrown
            index = randrange(len(total_colors))
            color = total_colors[index]
            hit_list.append(index)

            if color == 0:
                hits += 1

    area = (hits / darts) * 6

    return area, hit_list


def compare_area(iterations, darts):
    area_is = get_area(make_mandelbrot(iterations), darts)[0]

    compare_list = []
    j_list = []
    area_js_list = []
    for j in range(iterations - 1):
        iterations = j + 1
        area_js = get_area(make_mandelbrot(j + 1), darts)[0]
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
        area_i = get_area(make_mandelbrot(i + 1), darts)[0]
        area_list.append(area_i)

    plt.xlabel("Number of iterations")
    plt.ylabel("Area_i,s")
    plt.plot(range(1, max_iterations), area_list)
    plt.show()


def compare_s(iterations, max_darts):

    area_list = []
    for s in range(max_darts - 1):
        area_i = get_area(make_mandelbrot(iterations), s + 1)[0]
        # area_i = get_area_ortho(make_mandelbrot(s + 1), s + 1)[0]
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

def show_samples(total_colors, hit_list):
    """
    Return a list of colors in the Mandelbrot set.
    """
    im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    color_count = 0
    drawn_list = []
    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):

            # plot the point
            color = total_colors[color_count]
            if color_count in hit_list:
                draw.point([x, y], (255, 0, 0))

                # also color surrounding pixel to make samples more visible
                xlist = [x - 1, x + 1]
                ylist = [y - 1, y + 1]
                drawn_list = []
                for xsample in xlist:
                    for ysample in ylist:
                        draw.point([xsample, ysample], (255, 0, 0))
                        drawn_list.append([xsample, ysample])

            else:
                if [x, y] not in drawn_list:
                    draw.point([x, y], (color, color, color))
            color_count += 1

    im.show()

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


def calculate_variance(n, darts):
    """
    Calculate variance of all three methods.
    """

    # n = ((1.96/0.05)^2) * ((area/1.506)-1)

    methods = ["pure", "LHS", "ortho"]

    vars, means = [], []

    for method in methods:
        # total = []
        # for i in range(min, max):
        #     total.append(compare_s(i, 10))
        areas = compare_s(n, darts)
        vars.append(np.var(areas))
        means.append(np.mean(areas))
        print(method, "var", round(np.var(areas), 2), "mean", round(np.mean(areas), 2))

    return vars, means


def make_barplot(vars, means):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    methods = ["pure", "LHS", "ortho"]
    width = 0.27
    ticks = 3
    ind = np.arange(ticks)

    yvals = vars
    rects1 = ax.bar(ind, yvals, width, color='r')
    zvals = means
    rects2 = ax.bar(ind+width, zvals, width, color='g')

    ax.set_ylabel('Scores')
    ax.set_xticks(ind)
    ax.set_xticklabels(methods)
    ax.legend((rects1[0], rects2[0]), ('Variance', 'Mean'))

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*h, round(h,3),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)
    plt.title("n = 100, darts = 100")

    plt.show()

def make_linegraph():
    methods = ["pure", "LHS", "ortho"]
    iterations = 10
    darts = 10
    totalvars, totalmeans, iteration = [], [], []
    for i in range(1, 10):
        vars, means = calculate_variance(iterations, darts)
        totalvars.append(vars)
        totalmeans.append(means)
        iteration.append(i)
        iterations += 10
        darts += 50

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



if __name__ == '__main__':

    # # Barplot :)
    # iterations = 100
    # darts = 100
    # vars, means = calculate_variance(iterations, darts)
    # make_barplot(vars, means)

    # make_linegraph()

    # show samples in plot 
    total_colors = make_mandelbrot(20)
    hit_list = get_area_ortho(total_colors, 100, 2)[1]
    show_samples(total_colors, hit_list)
