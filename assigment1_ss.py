from PIL import Image, ImageDraw
import random
import matplotlib.pylab as plt
import numba

import numpy as np

# @numba.jit(nopython=True, parallel=True)


# @numba.jit(nopython=True, parallel=True)
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

    # throw darts
    for i in range(darts):

        # check if it landed in the mandelbrot
        color = random.choice(total_colors)

        # count = 0
        # for j in total_colors:
        #     if j != 0:
        #         count += 1
        # print(count, "ZO VAAK NIET NUL")
        # print(len(total_colors))

        if color == 0:
            hits += 1

    area = (hits / darts) * 6

    # moet 1.507 zijn

    return area


def get_area_lhs(total_colors, darts):
    """
    Throw darts darts in the domain and count how many hit the mandelbrot.
    Get random numbers through the LHS method.
    """
    hits = 0

    # Get list of latin-hybercube sampled values in 2 dimensions (X and Y)
    lhd = lhs(2, samples=darts)

    total_colors = np.array(total_colors).reshape(HEIGHT, WIDTH)

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
        # print("Point", total_colors[y][x])
        color = total_colors[y][x]

        if color == 0:
            hits += 1

    area = (hits / darts) * 6

    # moet 1.507 zijn

    return area


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


def compare_s(iterations, max_darts):

    area_list = []
    for s in range(max_darts - 1):
        area_i = get_area(make_mandelbrot(iterations), s + 1)
        area_list.append(area_i)

    plt.xlabel("Number of iterations")
    plt.ylabel("Area_i,s")
    plt.plot(range(1, max_darts), area_list)
    area_line = []
    print(len(area_list))
    for i in range(len(area_list)):
        area_line.append(1.507)
    plt.plot(range(1, max_darts), area_line)
    plt.show()


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


if __name__ == '__main__':
    # make_plot()
    # compare_area(1000, 700)
    # compare_i(100, 70)
    # make_mandelbrot(100)

    # compare_s(100, 100)

    compare_s(500, 300)
