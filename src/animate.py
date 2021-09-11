import imageio
import glob
def animate(case, length, index):
    filenames = sorted(glob.glob('../Trajectories/' + str(case) + '/' + str(length) + '/temp/' + '*.png'))
    images = []
    for i, filename in enumerate(filenames):
        images.append(imageio.imread(filename))
    imageio.mimsave('../Trajectories/' + str(case) + '/' + str(length) + '/' +
                     str(case) + '_' + str(length) + '_' + '%02d' % (index,) + '.gif', images, fps=10)