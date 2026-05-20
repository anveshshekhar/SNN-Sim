import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

print("="*40)
print("     BALL & BEAM SNN RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

b_targ = get_in("Ball Target Pos (m)", 0.0)
b_init = get_in("Ball Initial Pos (m)", 0.5)

print("-" * 40)
print("Initializing simulation buffers...")

G = 9.81
J_b, m_b, r_b, L_b = 2e-6, 0.05, 0.015, 1.0
K_acc = 5.0 / 7.0

T_MAX = 16.0
FPS = 60
DT_F = 1.0 / FPS
t_eval = np.arange(0, T_MAX, DT_F)
N = len(t_eval)
S_STEPS = 25
DT_P = DT_F / S_STEPS

N_S, N_R = 64, 300
np.random.seed(42)

beta = np.random.uniform(0.94, 0.99, N_R)
v = np.zeros(N_R)
W_in = np.random.uniform(-0.5, 0.5, (N_R, N_S))
W_res = np.random.randn(N_R, N_R) * 0.05
W_res[np.random.rand(*W_res.shape) > 0.2] = 0.0
W_out = np.zeros(N_R)
P_mat = np.eye(N_R) * 0.1

def get_spk(r, dr, alpha, dalpha):
    st = [r, dr, alpha, dalpha]
    cur = []
    for i, v_val in enumerate(st):
        lin = np.linspace(-1.0, 1.0, 16)
        c = (lin ** 3) * [1.0, 4.0, 0.5, 3.0][i]
        cur.extend(np.exp(-((c - v_val) ** 2) / (2 * 0.08 ** 2)))
    return np.array(cur)

h_r, h_alpha, h_snn = np.zeros(N), np.zeros(N), np.zeros(N)
r, dr, alpha, dalpha = b_init, 0.0, 0.0, 0.0
int_e = 0.0

for f in range(N):
    h_r[f], h_alpha[f] = r, alpha
    snn_sum = 0.0
    
    for sub in range(S_STEPS):
        err = b_targ - r
        int_e = np.clip(int_e + err * DT_P, -2.0, 2.0)
        
        spk = get_spk(r, dr, alpha, dalpha)
        inj = np.dot(W_in, spk) + np.dot(W_res, (v >= 0.2).astype(float))
        v = (beta * v) + inj
        act = (v >= 0.2).astype(float)
        v[v >= 0.2] -= 0.2
        
        f_snn = np.dot(W_out, act) * 10.0
        snn_sum += f_snn
        
        u = 15.0 * err - 6.0 * dr + 4.0 * int_e
        t_teach = -12.0 * alpha - 4.0 * dalpha + u
        
        error = f_snn - t_teach
        k_v = np.dot(P_mat, act)
        den = 1.0 + np.dot(act, k_v)
        if den > 1e-6:
            k_v /= den
            P_mat -= np.outer(k_v, np.dot(act, P_mat))
            W_out -= k_v * (error / 10.0)
            
        t_motor = np.clip(f_snn, -40.0, 40.0)
        ddalpha = (t_motor - 0.1 * dalpha) / 0.02
        dalpha += ddalpha * DT_P
        alpha += dalpha * DT_P
        alpha = np.clip(alpha, -0.4, 0.4)
        
        ddr = K_acc * (r * dalpha**2 - G * np.sin(alpha))
        dr += ddr * DT_P
        r += dr * DT_P
        
    h_snn[f] = snn_sum / S_STEPS

x_b, y_b = h_r * np.cos(h_alpha), h_r * np.sin(h_alpha)

fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(3, 2, width_ratios=[1.2, 1.0])

ax_anim = fig.add_subplot(gs[:, 0])
ax_anim.set_xlim(-0.7, 0.7)
ax_anim.set_ylim(-0.4, 0.4)
ax_anim.set_aspect('equal')
ax_anim.grid(True, linestyle='--', alpha=0.3)
ax_anim.set_title("Ball and Beam System View", fontsize=10, fontweight='bold')
beam_line, = ax_anim.plot([], [], 'o-', lw=6, color='#7f7f7f', markersize=4)
ball_mark, = ax_anim.plot([], [], 'o', color='#d62728', markersize=16)

W_WIN = 1.5
ax_wfr = fig.add_subplot(gs[0, 1])
ax_wfr.grid(True, linestyle='--', alpha=0.3)
ax_wfr.set_title("SNN Diagnostic Scope (Zoomed 1.5s Window)", fontsize=10, fontweight='bold')
ax_wfr.set_ylabel("Ball Pos (m)", fontsize=8)
ax_wfr.set_ylim(-0.6, 0.6)
ax_wfr.axhline(b_targ, color='r', linestyle='--', alpha=0.5)
tr_r, = ax_wfr.plot([], [], color='#1f77b4', lw=2)

ax_wfa = fig.add_subplot(gs[1, 1])
ax_wfa.grid(True, linestyle='--', alpha=0.3)
ax_wfa.set_ylabel("Beam Ang (rad)", fontsize=8)
ax_wfa.set_ylim(-0.5, 0.5)
tr_a, = ax_wfa.plot([], [], color='#2ca02c', lw=2)

ax_wfs = fig.add_subplot(gs[2, 1])
ax_wfs.grid(True, linestyle='--', alpha=0.3)
ax_wfs.set_ylabel("SNN Torque (N*m)", fontsize=8)
ax_wfs.set_xlabel("Time (s)", fontsize=8)
ax_wfs.set_ylim(np.min(h_snn) - 5, np.max(h_snn) + 5)
tr_s, = ax_wfs.plot([], [], color='#e377c2', lw=1.5)

ax_wfr.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfa.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfs.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
plt.tight_layout()

def init():
    beam_line.set_data([], [])
    ball_mark.set_data([], [])
    tr_r.set_data([], [])
    tr_a.set_data([], [])
    tr_s.set_data([], [])
    return beam_line, ball_mark, tr_r, tr_a, tr_s

def animate(i):
    t_c = t_eval[i]
    cos_a, sin_a = np.cos(h_alpha[i]), np.sin(h_alpha[i])
    beam_line.set_data([-0.5 * cos_a, 0.5 * cos_a], [-0.5 * sin_a, 0.5 * sin_a])
    ball_mark.set_data([x_b[i]], [y_b[i]])
    
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + DT_F
    t_e = W_WIN if t_c < W_WIN else t_c + DT_F
    ax_wfr.set_xlim(t_s, t_e)
    ax_wfa.set_xlim(t_s, t_e)
    ax_wfs.set_xlim(t_s, t_e)
    
    tr_r.set_data(t_eval[:i], h_r[:i])
    tr_a.set_data(t_eval[:i], h_alpha[:i])
    tr_s.set_data(t_eval[:i], h_snn[:i])
    return beam_line, ball_mark, tr_r, tr_a, tr_s

print("Starting display panels...")
ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=DT_F * 1000, blit=True, repeat=False)
plt.show()