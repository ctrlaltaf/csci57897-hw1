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


def pop_growth_SIR(S0, I0, R0, beta, gamma, t_max, stepsize, N_growth):
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

        N += N_growth

    return S, I, R, T


def SIR_birth_death(S0, I0, R0, beta, gamma, mu, t_max, stepsize):
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
            dS_dt = -beta * S[idx - 1] * I[idx - 1] / N - mu * S[idx - 1] + mu * N
            dI_dt = (
                beta * S[idx - 1] * I[idx - 1] / N
                - gamma * I[idx - 1]
                - mu * I[idx - 1]
            )
            dR_dt = gamma * I[idx - 1] - mu * R[idx - 1]

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


def r_func(x, r0):
    return 1 - np.exp(-r0 * x)


def main():
    print("hw1")
    base_output_path = Path("output/sir_fig.pdf")

    # plt from class
    S, I, R, T = base_SIR(999, 1, 0, 1, 0.5, 50, 0.05)
    plt.figure(figsize=(10, 8))
    plt.plot(T, S, color="b", label="Susceptibles")
    plt.plot(T, I, color="r", label="Infecteds")
    plt.plot(T, R, color="k", label="Recovereds")
    plt.xlabel("time")
    plt.ylabel("people")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(base_output_path)
    # plt.show()
    plt.close()

    p_growth_output_path = Path("output/p_growth_sir_fig.pdf")
    # SIR w/ population growth
    S, I, R, T = pop_growth_SIR(999, 1, 0, 1, 0.5, 50, 0.05, 2)
    plt.figure(figsize=(10, 8))
    plt.plot(T, S, color="b", label="Susceptibles")
    plt.plot(T, I, color="r", label="Infecteds")
    plt.plot(T, R, color="k", label="Recovereds")
    plt.xlabel("time")
    plt.ylabel("people")
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.savefig(p_growth_output_path)
    # plt.show()
    plt.close()

    # sir + b/d
    sir_bd_output = Path("output/sir_bd.pdf")
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    beta = 1
    gamma = 0.5
    mu = 0.01
    max_t = 500

    ax = axs[0]
    S, I, R, T = base_SIR(999, 1, 0, beta, gamma, max_t, 0.05)
    # ax.plot(T,S, color='b', label='Susceptibles')
    ax.plot(T, I, color="r", label="Infecteds")
    # ax.plot(T,R, color='k', label='Recovereds')

    ax = axs[1]
    S, I, R, T = SIR_birth_death(999, 1, 0, beta, gamma, mu, max_t, 0.05)
    # ax.plot(T,S, color='b', label='Susceptibles')
    ax.plot(T, I, color="r", label="Infecteds")
    # ax.plot(T,R, color='k', label='Recovereds')
    R0 = beta / (gamma + mu)
    Seq = 1 / R0 * 1000
    # ax.plot([0,max_t],[Seq,Seq],'b--')

    for ax in axs:
        ax.set_xlabel("time")
        ax.set_ylabel("people")
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0)
    plt.savefig(sir_bd_output)
    # plt.show()
    plt.close()

    # hw sir + b/d

    p_growth__sir_output_path = Path("output/hw_fig.pdf")
    # SIR w/ population growth
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
    plt.plot(T, I, color="r", label="Infecteds")
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
    plt.savefig(p_growth__sir_output_path)
    # plt.show()
    plt.close()

    # question 3.b & d

    # plt from class
    t_max = 10
    S, I, R, T = base_SIR(999, 1, 0, 1, 0.5, t_max, 0.05)
    S = S / N
    I = I / N
    R = R / N

    # plt.figure(figsize=(10, 8))
    # plt.plot(T, S, color="b", label="Susceptibles")
    # plt.plot(T, I, color="r", label="Infecteds")
    # plt.plot(T, R, color="green", label="Recovereds")

    x = np.linspace(0, t_max, 500)
    params = [0.9, 1.0, 1.1, 1.2]
    cumulative_incidence = Path("output/hw3_b.png")
    fig, axes = plt.subplots(len(params), 1, figsize=(6, 8), sharex=True)

    for ax, p in zip(axes, params):
        y = r_func(x, p)
        intersection = fsolve(lambda r: r_func(r, p) - r, 0.5)[0]
        ax.plot(x, y, label=f"param={p}", color="red")
        ax.plot(x, x, label="f(r_inf)", color="black")
        ax.plot(T, R, color="green", linestyle="--", label="Recovereds")
        ax.scatter(
            intersection,
            r_func(intersection, p),
            color="blue",
            s=60,
            zorder=5,
            label="intersection",
        )

        ax.set_ylabel("f(x)")
        ax.set_title(f"param={p}")
        ax.legend()
        ax.grid(True)

    axes[-1].set_xlabel("x")
    plt.tight_layout()
    plt.savefig(cumulative_incidence)
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

    q4_path = Path("output/q4.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(T_1, S_1, color="b", label="Susceptibles")
    ax1.plot(T_1, I_1, color="r", label="Infecteds")
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
    ax2.plot(T_2, I_2, color="r", label="Infecteds")
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
    plt.show()


if __name__ == "__main__":
    main()
