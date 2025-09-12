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

    p_growth__sir_output_path = Path("output/sir_p_growth_sir_fig.pdf")
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
    plt.plot(T, population, color="purple", label="Current Population", linestyle=':')
    plt.xlabel("time")
    plt.ylabel("people")
    plt.ylim(0, 1700)
    plt.legend()
    plt.ylim(bottom=0)
    plt.xlim(left=0)
    plt.yticks()
    plt.savefig(p_growth__sir_output_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
