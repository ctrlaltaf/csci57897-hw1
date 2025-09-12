from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


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
    plt.show()
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
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
