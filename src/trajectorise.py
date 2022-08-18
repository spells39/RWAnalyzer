import matplotlib.pyplot as plt
import numpy as np
import glob
import os
def trajectorise(game, case, length, index):
    fig, axes = plt.subplots(1, figsize=(8, 8), dpi=100)   
    axes.set_yticks(np.arange(1, 16, 1))
    axes.set_xticks(np.arange(1, 16, 1))
    axes.set_xlim(0, 16)
    axes.set_ylim(0, 16)
    axes.set_yticklabels([])
    axes.set_xticklabels([])
    axes.grid()
    axes.invert_yaxis()
    
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
        axes.set_title(str(i))
        axes.plot(x_values, y_values, 'b-')
        dot = new_position.pop(0)
        dot.remove()
        new_position = axes.plot(game[i][0], game[i][1], 'bo')
        plt.savefig('../Trajectories/' + str(case) + '/' + str(length) + '/temp/' + 
                     str(case) + '_' + str(length) + '_' + '%04d' % (i,) + '.png')
    plt.savefig('../Trajectories/' + str(case) + '/' + str(length) + '/' +
                     str(case) + '_' + str(length) + '_' + '%02d' % (index,) + '.png')
    axes.clear()