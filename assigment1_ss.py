import math
import random
import time
import argparse
import lhsmdu
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

iterations = 10000
samples = 1000


def _apfelrekursion(c, max_recursion):
    z = 0 + 0j
    zehn = range(9)
    while max_recursion > 0:
                max_recursion = max_recursion - 9
                #10 recursions at a time give a reasonable speed up. Doing more might end up in overflow errors.
                for x in zehn:
                    z = z * z + c
                if (z.real > 2) | (z.real < -2) | (z.imag > 2)|(z.imag < -2):
                    break
    return z


def isinmandelbrot(c, max_recursion, iterations):
    '''
    checks if complex value z is part of the Mandelbrot set or not. Limit number of max recursions approx. to max_recursion.
    Recursion:
    z_0 = 0
    z_{n+1} = z_n^2 + c
    Complex value c is part of Mandelbrot_set if abs((z_n)_n) < inf for n approaching inf.
    Find more at: http://en.wikipedia.org/wiki/Mandelbrot_set
    '''
    # cneu = _apfelrekursion(c, max_recursion)
    cneu = _apfelrekursion(c, max_recursion, iterations)
    return (cneu.real < 2) & (cneu.real > -2) & (cneu.imag < 2) & (cneu.imag > -2)


def mandelbrot_area_mc(N, iterations):
    '''
    Performs Monte Carlo Integration to calculate Mandelbrot Sets Area using N samples.
    Returns: (Area, Number of samples)
    '''
    ret = 0
    n = N
    while n > 0:
        n = n - 1
        # Dont want to assume symmetry

        if isinmandelbrot(complex(random.uniform(-2, 0.5), random.uniform(-1, 1)), N, iterations):
            ret += 1

    # Area is ret * (Area in cmplx plane the random numbers are covering)
    return float(ret) * 5 / N, ret

# def mandelbrot_area_mc(N):
#     '''
#     Performs Monte Carlo Integration to calculate Mandelbrot Sets Area using N points.
#     Returns: (Area, Number of points)
#     '''
#     ret = 0
#     n = N
#     while n > 0:
#         n = n - 1
#         #Dont want to assume symmetry
#         if isinmandelbrot(complex(random.uniform(-2, 0.5), random.uniform(-1, 1))):
#             ret += 1
#     #Area is ret * (Area in cmplx plane the random numbers are covering)
#     return float(ret) * 5 / N, ret

def comparable_test(iterations, samples):
    """
    test if i and s are comparable
    (for all j < i we compute Aj,s - Ai,s)
    """

    area_is = mandelbrot_area_mc(samples)

    for i in range(iterations):
        j = iterations - i


if __name__ == '__main__':

    total = []
    samples_total = []
    iterations_total = []

    for x in range(10):

        iterations = iterations / 2
        samples = samples / 2
        iterations_total.append(iterations)
        samples_total.append(samples)

        parser = argparse.ArgumentParser(description='''Calculating the Mandelbrot Set Area in Complex plane using Monte Carlo integration.
        Unconfirmed theory guesses the Area to be sqrt(6*pi - 1) - e = 1.506591...
        See also: http://en.wikipedia.org/wiki/Mandelbrot_set''')
        parser.add_argument('-n', type=float, help='Number of samples to use for Monte Carlo integration.', default=10000)
        args = parser.parse_args()

        N = int(args.n)
        print('Starting Monte Carlo using %s samples...' % N)
        startzeit = time.time()
        area, A = mandelbrot_area_mc(samples, iterations)
        std = (5 * math.sqrt(float(A)) / N)
        print('Area:        ' + str(area))
        print('Area StdDev: ' + str(std))
        print('Time (sec):  ' + str(time.time() - startzeit))

        total.append(std)


    plt.plot(iterations_total, total)
    plt.show()

    plt.plot(samples_total, total)
    plt.show()
