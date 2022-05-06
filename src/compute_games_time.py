import numpy as np

def compute_games_time(turns_timestamps):
    """
    This function computes whole time used by players.
    turns_timestamps --- the list of pandas series of turn timestamps for each game
    """
    all_time = 0
    for cur in turns_timestamps:
        time_diffs = (cur - cur.shift()).dt.seconds
        all_time += np.sum(time_diffs[time_diffs < 300])
    return all_time