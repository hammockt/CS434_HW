"""
Implementation for part 3.1 of the assignment
"""

import sys
import numpy
import matplotlib.pyplot as plt


def show_img(img):
    new_img = numpy.reshape(img, (28, 28))
    plt.imshow(new_img, cmap='gray', vmin=0, vmax=255)
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
    covar = numpy.cov(training_data, bias=True)
    es, evs = numpy.linalg.eigh(covar)
    es = numpy.flip(es)
    evs = numpy.flip(evs)
    print(es[:10])

main(sys.argv)
