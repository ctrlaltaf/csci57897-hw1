from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def base_SIR(S0, I0, R0, beta, gamma, t_max, stepsize):
    T = np.arange(0, t_max + stepsize, stepsize)
    S = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    N = S0 + I0 + R0

    for idx, t in enumerate(T):
        if idx == 0:
            S[idx] = S0
            I[idx] = I0
            R[idx] = R0
        else:
            dS_dt = -beta * S[idx - 1] * I[idx - 1] / N
            dI_dt = beta * S[idx - 1] * I[idx - 1] / N - gamma * I[idx - 1]
            dR_dt = gamma * I[idx - 1]

            S[idx] = S[idx - 1] + dS_dt * stepsize
            I[idx] = I[idx - 1] + dI_dt * stepsize
            R[idx] = R[idx - 1] + dR_dt * stepsize

    return S, I, R, T


def growth_SIR_birth_death(
    S0, I0, R0, beta, gamma, birth_rate, death_rate, t_max, stepsize, final_population
):
    T = np.arange(0, t_max + stepsize, stepsize)
    S = np.zeros(len(T))
    I = np.zeros(len(T))
    R = np.zeros(len(T))
    population = np.zeros(len(T))
    N = S0 + I0 + R0

    idx_stop = 0
    for idx, t in enumerate(T):
        if N < final_population:
            if idx == 0:
                S[idx] = S0
                I[idx] = I0
                R[idx] = R0
            else:
                dS_dt = (
                    -beta * S[idx - 1] * I[idx - 1] / N
                    - death_rate * S[idx - 1]
                    + birth_rate * N
                )
                dI_dt = (
                    beta * S[idx - 1] * I[idx - 1] / N
                    - gamma * I[idx - 1]
                    - death_rate * I[idx - 1]
                )
                dR_dt = gamma * I[idx - 1] - death_rate * R[idx - 1]

                S[idx] = S[idx - 1] + dS_dt * stepsize
                I[idx] = I[idx - 1] + dI_dt * stepsize
                R[idx] = R[idx - 1] + dR_dt * stepsize
                N = S[idx] + I[idx] + R[idx]
                population[idx] = N
        else:
            idx_stop = idx
            break

    T = T[0:idx_stop]
    S = S[0:idx_stop]
    I = I[0:idx_stop]
    R = R[0:idx_stop]
    population = population[0:idx_stop]

    return S, I, R, T, population


def r_equation(x, r0):
    return 1 - np.exp(-r0 * x)


def final_size_equation(r, r0):  # use this to solve equal to zero
    return r - (1 - np.exp(-r0 * r))


def main():
    print("hw1")

    # q1 sir + b/d w/ slow growth

    p_growth_sir_output_path = Path("output/hw_fig_q1.pdf")

    S0 = 999
    I0 = 1
    R0 = 0
    N = 1000
    beta = 1
    gamma = 0.5
    # mu = 0.01
    birth_rate = 0.01
    death_rate = 0.5 * birth_rate
    max_t = 50000
    final_population = 1500
    S, I, R, T, population = growth_SIR_birth_death(
        999, 1, 0, beta, gamma, birth_rate, death_rate, max_t, 0.05, final_population
    )
    plt.figure(figsize=(10, 8))
    plt.plot(T, S, color="b", label="Susceptibles")
    plt.plot(T, I, color="r", label="Infectious")
    plt.plot(T, R, color="k", label="Recovereds")
    plt.plot(T, population, color="purple", label="Current Population", linestyle=":")
    plt.xlabel("time")
    plt.ylabel("people")
    plt.ylim(0, 1700)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.yticks()
    plt.title(
        "Altaf Barelvi: HW1.Q1 Population dynamic of SIR model with population growth"
    )
    plt.savefig(p_growth_sir_output_path)
    # plt.show()
    plt.close()

    # question 3.b & d

    t_max = 0.5
    N = 1000
    S, I, R, T = base_SIR(999, 1, 0, 1, 0.5, t_max, 0.05)
    S = S / N
    I = I / N
    R = R / N

    x = np.linspace(0, t_max, 500)
    params = [0.9, 1.0, 1.1, 1.2]
    cumulative_incidence = Path("output/hw_fig_q3b.png")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, p in zip(axes, params):
        y = r_equation(x, p)
        intersection = fsolve(lambda r: r_equation(r, p) - r, 0.5)[0]

        ax.plot(x, y, label=f"param={p}", color="red")
        ax.plot(x, x, label="f(r_inf)", color="black")
        ax.scatter(
            intersection,
            r_equation(intersection, p),
            color="blue",
            s=60,
            zorder=5,
            label="intersection",
        )

        ax.set_title(f"param={p}")
        ax.set_ylabel("f(x)")
        ax.grid(True)
        ax.legend()

    for ax in axes[2:]:
        ax.set_xlabel("x")

    plt.tight_layout()
    plt.savefig(cumulative_incidence)
    # plt.show()
    plt.close()

    # question 3.d

    predicted_epi_size_path = Path("output/hw_fig_q3d")
    beta = 1
    gamma = 0.5
    R0 = beta / gamma
    r_inf_pred = fsolve(final_size_equation, 0.5, args=(R0,))[0]
    print("Predicted final epidemic size r_inf:", r_inf_pred)

    t_max = 100
    step_size = 0.05
    N = 1000

    S, I, R, T = base_SIR(999, 1, 0, beta, gamma, t_max, step_size)
    S = S / N
    I = I / N
    R = R / N

    plt.figure(figsize=(10, 8))
    plt.plot(T, S, color="b", label="Susceptibles")
    plt.plot(T, I, color="r", label="Infectious")
    plt.plot(T, R, color="g", label="Recovereds")

    plt.axhline(
        r_inf_pred,
        color="green",
        linestyle="--",
        label=f"r_inf prediction = {r_inf_pred:.3f}",
    )

    plt.xlabel("time")
    plt.ylabel("population (proportion)")
    plt.legend()
    plt.grid(True)
    plt.savefig(predicted_epi_size_path)
    # plt.show()
    plt.close()

    # question 4
    N = 10**6
    s0 = N - 1 / N
    i0 = 1 / N
    r0 = 0
    beta_1, gamma_1, t_max_1 = 1, 0.5, 100
    beta_2, gamma_2, t_max_2 = 1, 0.8, 500
    step_size = 0.005

    S_1, I_1, R_1, T_1 = base_SIR(s0, i0, r0, beta_1, gamma_1, t_max_1, step_size)
    S_1, I_1, R_1 = S_1 / N, I_1 / N, R_1 / N

    S_2, I_2, R_2, T_2 = base_SIR(s0, i0, r0, beta_2, gamma_2, t_max_2, step_size)
    S_2, I_2, R_2 = S_2 / N, I_2 / N, R_2 / N

    herd_immunity_threshold = gamma_1 / beta_1
    t_herd_threhold_reached = 0
    for s, t in zip(S_1, T_1):
        if s > herd_immunity_threshold:
            t_herd_threhold_reached = t

    x_max_i, y_max_i = 0, 0
    for i, t in zip(I_1, T_1):
        if i > y_max_i:
            x_max_i, y_max_i = t, i

    q4_path = Path("output/hw_fig_q4.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(T_1, S_1, color="b", label="Susceptibles")
    ax1.plot(T_1, I_1, color="r", label="Infectious")
    ax1.plot(T_1, R_1, color="k", label="Recovereds")
    ax1.axvline(
        x=t_herd_threhold_reached,
        color="blue",
        linestyle="--",
        linewidth=1,
        label=f"Herd threshold (s={herd_immunity_threshold:.2f})",
    )
    ax1.scatter(x_max_i, y_max_i, color="red", s=50, label="Peak of I")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Population (proportion)")
    ax1.set_title("SIR (β=1, γ=0.5, t_max=100)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(T_2, S_2, color="b", label="Susceptibles")
    ax2.plot(T_2, I_2, color="r", label="Infectious")
    ax2.plot(T_2, R_2, color="k", label="Recovereds")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Population (proportion)")
    ax2.set_title("SIR (β=1, γ=0.9, t_max=500)")
    ax2.legend()
    ax2.grid(True)

    fig.text(
        0.5,
        -0.02,
        "Figure showcasing two SIR simulations with different γ values",
        ha="center",
        fontsize=12,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(q4_path, bbox_inches="tight")
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
