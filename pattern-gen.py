import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

print("="*40)
print("     SNN CPG LOCOMOTION RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

f_drive = get_in("Drive Magnitude", 1.8)
t_sim   = get_in("Sim Duration (s)", 12.0)

print("-" * 40)
print("Initializing locomotion rhythmic buffers...")

FPS = 60
DT_F = 1.0 / FPS
t_eval = np.arange(0, t_sim, DT_F)
N = len(t_eval)

N_R = 4
tau_v, tau_u = 1.0, 5.0
beta, gamma = 2.5, 2.5

w = np.array([[ 0.0, -1.5,  0.0,  0.0],
              [-1.5,  0.0,  0.0,  0.0],
              [ 0.3,  0.0,  0.0, -1.5],
              [ 0.0,  0.3, -1.5,  0.0]])

v_res = np.zeros((N, N_R))
u_res = np.zeros((N, N_R))

for t in range(1, N):
    act = np.maximum(0.0, v_res[t-1, :])
    inj = f_drive + np.dot(w, act)
    
    v_res[t, :] = v_res[t-1, :] + (DT_F / tau_v) * (-v_res[t-1, :] - beta * u_res[t-1, :] + inj)
    u_res[t, :] = u_res[t-1, :] + (DT_F / tau_u) * (-u_res[t-1, :] + gamma * act)

h_out = np.maximum(0.0, v_res[:, 0]) - np.maximum(0.0, v_res[:, 1])

L_b = 1.0
x_p = L_b * np.sin(h_out)
y_p = -L_b * np.cos(h_out)

fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(3, 2, width_ratios=[1.2, 1.0])

ax_anim = fig.add_subplot(gs[:, 0])
ax_anim.set_xlim(-1.3, 1.3)
ax_anim.set_ylim(-1.3, 0.3)
ax_anim.set_aspect('equal')
ax_anim.grid(True, linestyle='--', alpha=0.3)
ax_anim.set_title("Locomotion Segment Gait Action", fontsize=10, fontweight='bold')
ax_anim.plot([-1.3, 1.3], [0, 0], 'k-', lw=3)
leg_line, = ax_anim.plot([], [], 'o-', lw=5, color='#2ca02c', markersize=10, markerfacecolor='#d62728')

W_WIN = 1.5
ax_wfn = fig.add_subplot(gs[0, 1])
ax_wfn.grid(True, linestyle='--', alpha=0.3)
ax_wfn.set_title("CPG Neural Rhythm Tracker (Zoomed 1.5s Window)", fontsize=10, fontweight='bold')
ax_wfn.set_ylabel("Node 1 Act", fontsize=8)
ax_wfn.set_xlim(0, W_WIN)
ax_wfn.set_ylim(-0.1, np.max(v_res[:, 0]) + 0.2)
tr_n, = ax_wfn.plot([], [], color='#1f77b4', lw=2)

ax_wfi = fig.add_subplot(gs[1, 1])
ax_wfi.grid(True, linestyle='--', alpha=0.3)
ax_wfi.set_ylabel("Node 2 Act", fontsize=8)
ax_wfi.set_xlim(0, W_WIN)
ax_wfi.set_ylim(-0.1, np.max(v_res[:, 1]) + 0.2)
tr_i, = ax_wfi.plot([], [], color='#2ca02c', lw=2)

ax_wfo = fig.add_subplot(gs[2, 1])
ax_wfo.grid(True, linestyle='--', alpha=0.3)
ax_wfo.set_ylabel("Gait Cmd Out", fontsize=8)
ax_wfo.set_xlabel("Time (s)", fontsize=8)
ax_wfo.set_xlim(0, W_WIN)
ax_wfo.set_ylim(np.min(h_out) - 0.2, np.max(h_out) + 0.2)
tr_o, = ax_wfo.plot([], [], color='#e377c2', lw=2)

ax_wfn.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfi.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfo.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
plt.tight_layout()

def init():
    leg_line.set_data([], [])
    tr_n.set_data([], [])
    tr_i.set_data([], [])
    tr_o.set_data([], [])
    return leg_line, tr_n, tr_i, tr_o

def animate(i):
    t_c = t_eval[i]
    leg_line.set_data([0, x_p[i]], [0, y_p[i]])
    
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + DT_F
    t_e = W_WIN if t_c < W_WIN else t_c + DT_F
    ax_wfn.set_xlim(t_s, t_e)
    ax_wfi.set_xlim(t_s, t_e)
    ax_wfo.set_xlim(t_s, t_e)
    
    tr_n.set_data(t_eval[:i], np.maximum(0.0, v_res[:i, 0]))
    tr_i.set_data(t_eval[:i], np.maximum(0.0, v_res[:i, 1]))
    tr_o.set_data(t_eval[:i], h_out[:i])
    return leg_line, tr_n, tr_i, tr_o

print("Starting display panels...")
ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=DT_F * 1000, blit=True, repeat=False)
plt.show()