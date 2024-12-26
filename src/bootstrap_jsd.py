from scipy.spatial.distance import jensenshannon
import numpy as np

def bootstrap_jsd(prob_real, prob_model, n_bootstrap=1000):
    """
    Вычисляет p-value с использованием бутстреппинга для JSD.
    """
    jsd_observed = jensenshannon(prob_real, prob_model)
    jsd_bootstrap = []

    for _ in range(n_bootstrap):
        bootstrap_real = np.random.choice(prob_real, size=len(prob_real), replace=True)
        bootstrap_model = np.random.choice(prob_model, size=len(prob_model), replace=True)
        
        jsd_bootstrap.append(jensenshannon(bootstrap_real, bootstrap_model))

    p_value = np.mean(np.array(jsd_bootstrap) >= jsd_observed)
    return p_value