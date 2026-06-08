import matplotlib.pyplot as plt
import numpy as np
import glob
import os


def _configure_field(axes, field_size=16):
    axes.set_xticks(np.arange(1, field_size, 1))
    axes.set_yticks(np.arange(1, field_size, 1))
    axes.set_xlim(0, field_size)
    axes.set_ylim(0, field_size)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.grid(True)
    axes.set_aspect("equal")
    axes.invert_yaxis()


def save_empty_field(output_path, field_size=16, figsize=(8, 8), dpi=100):
    fig, axes = plt.subplots(1, figsize=figsize, dpi=dpi)
    _configure_field(axes, field_size)
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="white",
    )
    plt.close(fig)


def trajectorise(game, case, length, index):
    fig, axes = plt.subplots(1, figsize=(8, 8), dpi=100)
    _configure_field(axes)
    
    files = glob.glob('../Trajectories/' + str(case) + '/' + str(length) + '/temp/*.png', recursive=True)
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))
        
    for i in range(1, len(game)):
        if i == 1:
            new_position = axes.plot(game[i-1][0], game[i-1][1], 'bo')
        x_values = []
        y_values = []
        x_values.append(game[i-1][0])
        x_values.append(game[i][0])
        y_values.append(game[i-1][1])
        y_values.append(game[i][1])
        axes.set_title(str(i) + "/" + str(len(game) - 1))
        axes.plot(x_values, y_values, 'b-')
        dot = new_position.pop(0)
        dot.remove()
        new_position = axes.plot(game[i][0], game[i][1], 'bo')
        plt.savefig('../Trajectories/' + str(case) + '/' + str(length) + '/temp/' + 
                     str(case) + '_' + str(length) + '_' + '%04d' % (i,) + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
        
    plt.savefig('../Trajectories/' + str(case) + '/' + str(length) + '/' +
                     str(case) + '_' + str(length) + '_' + '%02d' % (index,) + '.png', transparent=False, \
                facecolor='white', edgecolor='white')
    plt.close(fig)
