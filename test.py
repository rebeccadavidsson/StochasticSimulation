
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import seaborn as sns
from textwrap import wrap
sns.set()


def make_barplot_var(vars, title):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    width = 0.27
    ticks = 6
    ind = np.arange(ticks)

    yvals = vars
    rects1 = ax.bar(ind, yvals, width, color='r')
    # zvals = means
    # rects2 = ax.bar(ind+width, zvals, width, color='g')

    ax.set_ylabel('Scores', fontsize=20)
    ax.set_xticks(ind)
    labels = ["Pure", "Pure antithetic",  "LHS", "LHS antithetic", "orthogonal", "orthogonal antithetic" ]
    labels = [ '\n'.join(wrap(l, 10)) for l in labels ]
    ax.set_xticklabels(labels, fontsize=15)
    # ax.legend((rects1[0], rects2[0]), ('Variance', 'Mean estimated area'), loc="upper center", bbox_to_anchor=(0.5, -0.05),shadow=True, ncol=2)

    def autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.00*h, round(h, 4),
                    ha='center', va='bottom')

    autolabel(rects1)
    # autolabel(rects2)
    plt.title(title, fontsize=20)
    plt.show()



if __name__ == '__main__':


    # # Barplot :)
    totalmeans = [1.5190825183038676, 1.5832165114921837, 1.5564125819899013, 1.552142635901539, 1.558063249751437, 1.5586903417125932]
    newmeans = []
    for i in totalmeans:
        newmeans.append(i - 0.05)

    print(newmeans)

    make_barplot_var([0.031889449319120154, 0.017367776417237878, 0.016667351610450926, 0.0026529062350537373, 0.012504947981976572, 0.0014901522324351586], "Mean sample variance")
    make_barplot_var(newmeans, "Mean estimated area")
