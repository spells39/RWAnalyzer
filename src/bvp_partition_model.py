from bvp_quarter_model import solve_bvp_quarter_sweep


def compute_radius(n):
    return n // 4


def solve_bvp_sweep(
    n,
    epsilon_values,
    output_duration,
    output_absorption_images1,
    output_absorption_images2,
    output_absorption_images3,
    qr_matrices,
    **kwargs,
):
    return solve_bvp_quarter_sweep(
        epsilon_values=epsilon_values,
        output_duration=output_duration,
        output_absorption_images1=output_absorption_images1,
        output_absorption_images2=output_absorption_images2,
        output_absorption_images3=output_absorption_images3,
        qr_matrices=qr_matrices,
        n=n,
    )
