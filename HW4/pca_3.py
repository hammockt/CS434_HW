"""
Implementation for part 3.3 of the assignment
"""

import sys
import numpy
import matplotlib.pyplot as plt


def show_img(img, dims):
    fig = plt.figure()
    new_img = numpy.reshape(img, (28, 28))
    dim_img = numpy.reshape(dims, (2, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(new_img, cmap="gray", vmin=min(img), vmax=max(img))
    fig.add_subplot(1, 2, 2)
    plt.imshow(dim_img, cmap="gray", vmin=min(dims), vmax=max(dims))
    plt.show()

def main(argv):
    """ entry point """
    if len(argv) != 1:
        sys.exit(f"Usage: {argv[0]}")

    ################
    # get the data #
    ################

    training_data = numpy.genfromtxt("p4-data.txt", delimiter=",")

    mean_img = numpy.mean(training_data, axis=0)
    covar = numpy.cov(training_data, rowvar=False, bias=True)
    es, evs = numpy.linalg.eigh(covar)
    es = numpy.flip(es)
    # Need to get rows instead of columns
    evs = numpy.flip(evs.T)
    proj_matrix = (evs[:10]).T
    reduced_matrix = training_data @ proj_matrix
    for i in range(10):
        show_img(evs[i], reduced_matrix[i])

main(sys.argv)
