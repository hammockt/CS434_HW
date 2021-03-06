"""
Implementation for part 3.2 of the assignment
"""

import sys
import numpy
import matplotlib.pyplot as plt


def show_img(mean, img):
    fig = plt.figure()
    mean_img = numpy.reshape(mean, (28, 28))
    new_img = numpy.reshape(img, (28, 28))
    fig.add_subplot(1, 2, 1)
    plt.imshow(mean_img, cmap="gray", vmin=min(mean), vmax=max(mean))
    fig.add_subplot(1, 2, 2)
    plt.imshow(new_img, cmap='gray', vmin=min(img), vmax=max(img))
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
    for i in range(10):
        show_img(mean_img, evs[i])

main(sys.argv)
