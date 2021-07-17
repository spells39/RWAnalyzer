import imageio
import glob
def animate(case, length):
    filenames = sorted(glob.glob('../Trajectories/' + csae + '/' + length + '/temp/' + '*.png'))
    images = []
    for i, filename in enumerate(filenames):
        images.append(imageio.imread(filename))
    imageio.mimsave('../Trajectories/' + case + '/' + length + '/' +
                     case + '_' + length + '.gif', images, fps=10)