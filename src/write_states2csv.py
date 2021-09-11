import csv
import numpy as np
def write_states2csv(case, length, game, i):
    directory = "../Trajectories/" + case + "/" + length + "/"
    filename = case + "_" + length + "_" + str(i) + ".csv"
    with open(directory + filename, "w+") as my_csv:
        csvWriter = csv.writer(my_csv,delimiter=',')
        csvWriter.writerow("xy")
        csvWriter.writerows(game)