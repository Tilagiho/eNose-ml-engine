import funcdataset

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import multiprocessing
import errno
import time

# constants
source_dir = "data/eNose 200122 - Timo/Zelle/dataset_0403_full"
dest_dir_extension = "3d_60"
dest_dir = "images/" + source_dir.split("/")[-1] + "_" + dest_dir_extension

plot3d = False
useRelativeVectors = True
useFuncVectors = True

image_size = (224, 224)     # size of image in pixels
my_dpi = 96

extreme_factor = 1.1    # factor by which max & min will be multiplied
xmin = -2
xmax = 2
ymin = -2
ymax = 2


def generateImage(point_list, plot_3d=False):
    figure = plt.figure(figsize=(image_size[0] / my_dpi, image_size[1] / my_dpi), dpi=my_dpi)
#    ax = fig.add_subplot(111)
    ax = figure.add_subplot(111, frameon=False)
    plt.axis('off')

    # iterate through list in reverse order
    for (i, point) in enumerate(point_list[::2]):
        x = [point[0, 0]]
        y = [point[0, 1]]
        z = [point[0, 2]]
        if z[0] > 0.5 or z[0] < -0.5:
            print("---\n\n\nhere\n\n\n---")

        # create color
        value = 0.8 * (1 - i / len(point_list)) + 0.2
        rgb_color = np.array([[1-value, value, value]])

        # determine size
        if plot_3d:
            size = 1 + 3 * z / (zmax - zmin)
        else:
            size = 1
        ax.scatter(x, y, c=rgb_color, s=size, zorder=len(point_list)-i)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # figure.show()
    # try:
    #     figure.draw()
    # except TypeError:
    #     pass

    # plt.pause(1e-17)
    # plt.gcf().canvas.draw_idle()
    # plt.gcf().canvas.start_event_loop(1e-17)

    return figure


def saveImage(figure: plt.figure, filepathname: str):
    # create directories
    if not os.path.exists(os.path.dirname(filepathname)):
        try:
            os.makedirs(os.path.dirname(filepathname))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # write to image file
    with open(filepathname, 'wb+') as file:
        figure.savefig(file, dpi=my_dpi)


def image_gen_worker(queue: multiprocessing.Queue, plot3d: bool):
    while True:
        # check queue, sleep if queue is empty
        if queue.empty():
            time.sleep(0.5)
        else:
            item = queue.get()

            # check for stop signal
            if item == None:
                queue.put(None)
                return

            # otherwise: generate & save image
            (vector_list, timestamp, annotation, count, i_file, i_vector) = item

            print(f"{os.getpid()}: {i_file} \t-\t {i_vector}")

            # create filename
            filename = annotation + "_" + str(count) + "_" + timestamp + ".png"
            date = timestamp.split(" -")[0]
            filepathname = dest_dir + "/" + date + "/" + annotation + "/" + filename

            # normalize & transform vectors into pca points
            point_list = []
            for vector in vector_list:
                vector = scaler.transform(np.array([vector]))
                point_list.append(pca.transform(vector))

            figure = generateImage(point_list, plot3d)
            saveImage(figure, filepathname)
            plt.close(figure)


# main program
if __name__ == '__main__':
    # load data
    dataset = funcdataset.FuncDataset(source_dir, convertToRelativeVectors=useRelativeVectors, calculateFuncVectors=useFuncVectors)

    # calc pca based on full data
    # data is normalized -> expected to be largely in range (-1,1)
    (pca, scaler, max, min) = dataset.getPCA(useTrainset=False, nComponents=3)
    (xmax, ymax, zmax) = (extreme_factor*max[0], extreme_factor*max[1], extreme_factor*max[2])
    (xmin, ymin, zmin) = (extreme_factor*min[0], extreme_factor*min[1], extreme_factor*min[2])

    # init iterator
    measIter = funcdataset.MeasIterator(n=30, funcdataset=dataset)

    print("Generating images...")
    # prepare queue & workers
    queue = multiprocessing.Queue()

    workers = []
    n_cpus = len(os.sched_getaffinity(0))     # leave one cpu unused
    if n_cpus < 1:
        n_cpus = 1
    for i in range(n_cpus):
        p = multiprocessing.Process(target=image_gen_worker, args=(queue, plot3d))
        workers.append(p)
        p.start()

    # fill queue
    for image_data in measIter:
        # put image data into queue
        (vector_list, timestamp, annotation, annotation_n, i_file, i_vector) = image_data
        print(f"{i_file} - {i_vector}: {annotation}[{annotation_n}]")
        queue.put(image_data)

        # secure against filling queue too much
        if queue.qsize() > 10*n_cpus:
            time.sleep(2)

     # send end signal to workers
    queue.put(None)

    for worker in workers:
        worker.join()