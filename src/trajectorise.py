import matplotlib.pyplot as plt
def trajectorise(game, axes, case, length):
    fig, axes = plt.subplots(1, figsize=(8, 8))
    axes.grid()
    axes.set_yticks(range(0,17))
    axes.set_xticks(range(0,17))
    axes.set_yticklabels([])
    axes.set_xticklabels([])

    for i in range(game):
        x_values.append(game[i][0]).append(game[i+1][0])
        y_values.append(game[i][1]).append(game[i+1][1])
        axes.plot(x_values, y_values)
        axes.savefig('../Trajectories/' + case + '/' + length + '/temp/' + 
                     case + '_' + length + '_' + i + '.png')
    axes.clear()