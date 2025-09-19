from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import textwrap


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

    p_growth_sir_output_path = Path("output/hw_fig_q1.png")
    S0 = 999
    I0 = 1
    R0 = 0
    N = 1000
    beta = 1
    gamma = 0.5
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
    plt.xlabel("Time")
    plt.ylabel("People")
    plt.ylim(0, 1700)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.yticks()
    plt.title(
        "Altaf Barelvi: HW1.Q1 Population dynamic of SIR model with population growth"
    )
    plt.figtext(
        0.5,
        0.04,
        "The plot shows how the number of susceptible, infectious, and recovered people changes over time given some population growth.",
        ha="center",
        fontsize=10,
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
    print()
    print("Question 3.b")
    for ax, p in zip(axes, params):
        y = r_equation(x, p)
        intersection = fsolve(lambda r: r_equation(r, p) - r, 0.5)[0]
        print(
            f"r_0 = {p} intersection at ({intersection}, {r_equation(intersection, p)})"
        )
        ax.plot(x, y, label=f"g(r_inf) w/ param={p}", color="red")
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
        ax.set_ylabel("y")
        ax.grid(True)
        ax.legend()

    for ax in axes[2:]:
        ax.set_xlabel("x")

    fig.suptitle(
        "Altaf Barelvi: HW1.Q3.B Solutions to the Transcendental Equation for Different Parameters",
        fontsize=14,
        y=1.02,
    )

    # Add a caption under the figure
    figure_caption = (
        "The figure shows the intersections between f(r_inf) (red) and g(r_inf) (black) "
        "for various parameter values. Blue circles mark the fixed points r_inf, "
        "indicating the epidemic's final size for each R0."
    )
    wrapped_caption = "\n".join(textwrap.wrap(figure_caption, width=100))

    fig.text(
        0.5,
        -0.04,
        wrapped_caption,
        ha="center",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(cumulative_incidence, bbox_inches="tight")
    plt.close()

    print()
    print("Question 3.d")

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
    print("final size of the R population proportion : ", R[-1])

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

    # Add a global title
    plt.title(
        "Altaf Barelvi: HW1.Q3.D SIR Model Dynamics and Predicted Final Epidemic Size",
        fontsize=14,
        y=1.02,
    )

    # Add a caption under the figure
    figure_caption = (
        "The simulation shows the evolution of susceptibles (blue), infectious (red), and recovereds (green). "
        "The dashed line indicates the predicted final epidemic size (0.796), which closely matches the proportion "
        "of individuals in the recovered class at the end of the outbreak (0.798)."
    )
    wrapped_caption = "\n".join(textwrap.wrap(figure_caption, width=100))

    plt.figtext(
        0.5,
        -0.02,
        wrapped_caption,
        ha="center",
        fontsize=10,
    )

    plt.savefig(predicted_epi_size_path, bbox_inches="tight")
    plt.close()

    # question 4
    q4_path = Path("output/hw_fig_q4.png")

    N = 10**6
    S0, I0, R00 = N - 1, 1, 0
    t_max = 200
    dt = 0.1

    cases = [
        ("Sim_1 = Stable (R0 < 1)", 0.5, 1.0),  # R0 = 0.5
        ("Sim_2 = Threshold (R0 = 1)", 1.0, 1.0),  # R0 = 1
        ("Sim_3 = Unstable (R0 > 1)", 1.5, 1.0),  # R0 = 1.5
    ]

    figure_caption = (
        "The figure illustrates the stability of the disease-free equilibrium (DFE) in the SIR model "
        "under three parameter, using identical initial conditions (N = 10^6, S = N-1, "
        "I = epsilon = 1/N, R = 0). The infectious proportion i(t) is shown on a logarithmic scale. "
        "When R0 = beta/gamma < 1 (blue curve, beta=0.5, gamma=1.0), infections decline exponentially to zero without an increase, "
        "which means that the DFE is stable. At the threshold R0 = 1 (orange curve, beta=1.0, gamma=1.0), "
        "infections remain constant at their initial condition. Since R0 = 1, the rate of recovery and the infection trasmission rate is equal, which supports the horizontal line. "
        "When R0 > 1 (green curve, beta=1.5, gamma=1.0), infections initially grow before declining, "
        "showing that the DFE is unstable. Just like simulation 1, the proportion of the infectious population will eventually reach zero, however since R0 > 1 it became a disease "
        "endemic equilibrium since the infection was able to grow."
        "This demonstrates the principle that the DFE is stable "
        "if s < 1/R0 and unstable otherwise."
    )

    plt.figure(figsize=(10, 8))

    for label, beta, gamma in cases:
        S, I, R, T = base_SIR(S0, I0, R00, beta, gamma, t_max, dt)
        plt.plot(T, I / N, label=f"{label} (beta={beta}, gamma={gamma})")

    plt.yscale("log")  # log scale to make plt more visible
    plt.xlabel("Time")
    plt.ylabel("Infectious proportion, i(t) [log scale]")
    plt.title(
        "Altaf Barelvi: HW1.Q4 Stability of the Disease-Free Equilibrium (N = 10^6, epsilon = 1/N)"
    )
    plt.tight_layout(rect=[0, 0.2, 1, 1])
    wrapped_caption = "\n".join(textwrap.wrap(figure_caption, width=100))
    plt.figtext(0.5, 0.01, wrapped_caption, ha="center", fontsize=9)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.7)
    plt.savefig(q4_path)
    # plt.show()
    plt.close()


if __name__ == "__main__":
    main()
