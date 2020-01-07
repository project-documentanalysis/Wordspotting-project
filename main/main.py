import pickle as pickle
import PIL.Image as Image
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.cluster.vq import kmeans2


def patch_wordspotting():
    document_image_filename = '2700270.png'
    image = Image.open(document_image_filename)
    im_arr = np.asarray(image, dtype='float32')
    plt.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    selectSIFT(5, 5, im_arr)
    plt.show()

def selectSIFT(step_size, cell_size, im_arr):
    pickle_densesift_fn = '2700270-full_dense-%d_sift-%d_descriptors.p' % (step_size, cell_size)
    frames, desc = pickle.load(open(pickle_densesift_fn, 'rb'))

    n_centroids = 40
    _, labels = kmeans2(desc, n_centroids, iter=20, minit='points')

    #Kommentar

    draw_descriptor_cells = True
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im_arr, cmap=cm.get_cmap('Greys_r'))
    ax.autoscale(enable=False)
    colormap = cm.get_cmap('jet')
    desc_len = cell_size * 4
    for (x, y), label in zip(frames, labels):
        color = colormap(label / float(n_centroids))
        circle = Circle((x, y), radius=1, fc=color, ec=color, alpha=1)
        rect = Rectangle((x - desc_len / 2, y - desc_len / 2), desc_len, desc_len, alpha=0.08, lw=1)
        ax.add_patch(circle)
        if draw_descriptor_cells:
            for p_factor in [0.25, 0.5, 0.75]:
                offset_dyn = desc_len * (0.5 - p_factor)
                offset_stat = desc_len * 0.5
                line_h = Line2D((x - offset_stat, x + offset_stat), (y - offset_dyn, y - offset_dyn), alpha=0.08, lw=1)
                line_v = Line2D((x - offset_dyn, x - offset_dyn), (y - offset_stat, y + offset_stat), alpha=0.08, lw=1)
                ax.add_line(line_h)
                ax.add_line(line_v)
        ax.add_patch(rect)

    plt.show()

if __name__ == '__main__':
    patch_wordspotting()
