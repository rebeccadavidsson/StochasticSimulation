from PIL import Image, ImageDraw
import random
from pyDOE import lhs
import numpy as np
import matplotlib.pylab as plt
import numba


# @numba.jit(nopython=True, parallel=True)
def mandelbrot(c, max_iterations):
    z = 0
    n = 0
    while abs(z) <= 2 and n < max_iterations:
        z = z*z + c
        n += 1
    return n

#
# for a in range(-10, 10, 5):
#     for b in range(-10, 10, 5):
#         c = complex(a / 10, b / 10)
        # print(c, mandelbrot(c))


# Image size (pixels)
# WIDTH = 600
# HEIGHT = 400
WIDTH = 6
HEIGHT = 4

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1

palette = []
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
        # for j in range(len(total_colors)):
        #     if color != 0:
        #         count += 1

        if color == 0:
            hits += 1

    area = (hits / darts) * 6 / 1.5
    # area = hits / darts
    # print(hits, area)
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
    # area = hits / darts

    return area


def compare_area(iterations, darts, method):

    if method == "lhs":
        area_is = get_area_lhs(make_mandelbrot(iterations), darts)
    elif method == "pure":
        area_is = get_area(make_mandelbrot(iterations), darts)

    # print(area_is)
    # print("-------")

    # compare_list = []
    # for j in range(iterations - 2):
    #
    #     area_js = get_area(make_mandelbrot(j + 1), darts)
    #     difference = area_js - area_is
    #     compare_list.append(difference)

    return area_is # TODO


def make_mandelbrot(iterations):
    """
    Return a list of colors in the Mandelbrot set.
    """

    total_colors = []
    palette = []

    # im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    # draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):

            # Convert pixel coordinate to complex number
            c = complex(RE_START + (x / WIDTH) * (RE_END - RE_START),
                        IM_START + (y / HEIGHT) * (IM_END - IM_START))
            # c = complex(x, y)
            # Compute the number of iterations
            m = mandelbrot(c, iterations)

            # The color depends on the number of iterations
            color = 255 - int(m * 255 / iterations)

            total_colors.append(color)

    # im.show()

    # print(total_colors)
    return total_colors


def make_plot():
    """
    Plot the number of hits against number of iterations.
    """

    methods = ["pure", "lhs"]
    darts = 10

    for method in methods:
        total = []
        horizontal = []
        j = 20
        for i in range(1, 400):
            total.append(compare_area(i, j, method))
            if j % 5 == 0:
                j = j + 10
            horizontal.append(1.509)
        print(total)
        print(np.mean(total))

        plt.title(method)
        plt.ylabel("Area ")
        plt.xlabel("Number of iterations")
        plt.plot(total)
        plt.plot(horizontal)
        plt.show()


if __name__ == '__main__':
    make_plot()
