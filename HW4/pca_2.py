"""
Implementation for part 3 of the assignment
"""

import sys
import numpy
import matplotlib.pyplot as plt


def show_img(img):
    new_img = numpy.reshape(img, (28, 28))
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
    #show_img(mean_img)
    covar = numpy.cov(training_data, rowvar=False, bias=True)
    es, evs = numpy.linalg.eigh(covar)
    es = numpy.flip(es)
    # Need to get rows instead of columns
    evs = numpy.flip(evs.T)
    #print(es[:10])
    for i in range(10):
        show_img(mean_img)
        show_img(evs[i])

main(sys.argv)
