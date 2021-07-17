from pathlib import Path
def make_directories(case):
    p1 = Path("../Trajectories/" + case + "/short")
    p2 = Path("../Trajectories/" + case + "/regular")
    p3 = Path("../Trajectories/" + case + "/long")
    p1.mkdir(parents=True, exist_ok=True)
    p2.mkdir(parents=True, exist_ok=True)
    p3.mkdir(parents=True, exist_ok=True)