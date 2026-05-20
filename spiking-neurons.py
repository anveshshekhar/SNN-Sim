import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

print("="*40)
print("     NEURON MODELS RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

i_inj = get_in("Injected Current (pA)", 7.5)
t_sim = get_in("Sim Time (ms)", 200.0)

print("-" * 40)
print("Initializing neuron models...")

dt = 0.1
t_eval = np.arange(0, t_sim, dt)
N = len(t_eval)

v_lif, ref_lif = np.zeros(N), np.zeros(N)
v_lif[0] = -70.0
v_th_lif, v_reset_lif, r_m, tau_m = -50.0, -70.0, 5.0, 20.0

v_izh, u_izh = np.zeros(N), np.zeros(N)
v_izh[0], u_izh[0] = -65.0, -14.0
a, b, c, d = 0.02, 0.2, -65.0, 6.0

h_spk_lif, h_spk_izh = np.zeros(N), np.zeros(N)

for t in range(1, N):
    if ref_lif[t-1] > 0:
        v_lif[t] = v_reset_lif
        ref_lif[t] = ref_lif[t-1] - dt
    else:
        v_lif[t] = v_lif[t-1] + (dt / tau_m) * (-(v_lif[t-1] - v_reset_lif) + r_m * i_inj)
        if v_lif[t] >= v_th_lif:
            v_lif[t] = 0.0
            ref_lif[t] = 2.0
            h_spk_lif[t] = 1.0

    v_prev = v_izh[t-1]
    v_izh[t] = v_prev + dt * (0.04 * v_prev**2 + 5.0 * v_prev + 140.0 - u_izh[t-1] + i_inj)
    u_izh[t] = u_izh[t-1] + dt * a * (b * v_prev - u_izh[t-1])
    if v_izh[t] >= 30.0:
        v_izh[t] = c
        u_izh[t] += d
        h_spk_izh[t] = 1.0

W_WIN = 30.0
fig = plt.figure(figsize=(14, 7))
gs = GridSpec(2, 1)

ax_lif = fig.add_subplot(gs[0, 0])
ax_lif.grid(True, linestyle='--', alpha=0.3)
ax_lif.set_title("Leaky Integrate-and-Fire (LIF) Dynamics", fontsize=10, fontweight='bold')
ax_lif.set_ylabel("Voltage (mV)", fontsize=8)
tr_lif, = ax_lif.plot([], [], color='#1f77b4', lw=2)
ax_lif.set_ylim(-75, 10)
ax_lif.xaxis.set_major_locator(ticker.MultipleLocator(5.0))

ax_izh = fig.add_subplot(gs[1, 0])
ax_izh.grid(True, linestyle='--', alpha=0.3)
ax_izh.set_title("Izhikevich Spiking Dynamics", fontsize=10, fontweight='bold')
ax_izh.set_ylabel("Voltage (mV)", fontsize=8)
ax_izh.set_xlabel("Time (ms)", fontsize=8)
tr_izh, = ax_izh.plot([], [], color='#e377c2', lw=2)
ax_izh.set_ylim(-75, 40)
ax_izh.xaxis.set_major_locator(ticker.MultipleLocator(5.0))

plt.tight_layout()

def init():
    tr_lif.set_data([], [])
    tr_izh.set_data([], [])
    return tr_lif, tr_izh

def animate(i):
    t_c = t_eval[i]
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + dt
    t_e = W_WIN if t_c < W_WIN else t_c + dt
    
    ax_lif.set_xlim(t_s, t_e)
    ax_izh.set_xlim(t_s, t_e)
    
    tr_lif.set_data(t_eval[:i], v_lif[:i])
    tr_izh.set_data(t_eval[:i], v_izh[:i])
    return tr_lif, tr_izh

ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=dt*100, blit=False, repeat=False)
plt.show()