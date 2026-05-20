import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

print("="*40)
print("     PENDULUM SNN ENGINE RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

th1_targ = get_in("Theta 1 (rad)", 0.35)
th2_targ = get_in("Theta 2 (rad)", -0.2)

print("-" * 40)
print("Initializing simulation buffers...")

G = 9.81
L1, L2 = 1.0, 1.0
M1, M2 = 1.0, 1.0

T_MAX = 16.0
FPS = 60
DT_F = 1.0 / FPS
t_eval = np.arange(0, T_MAX, DT_F)
N = len(t_eval)
S_STEPS = 25  
DT_P = DT_F / S_STEPS  

N_S, N_R = 64, 400
np.random.seed(101)

beta = np.random.uniform(0.93, 0.99, N_R)
v_res = np.zeros(N_R)
W_in = np.random.uniform(-0.4, 0.4, (N_R, N_S))
W_res = np.random.randn(N_R, N_R) * 0.04
W_res[np.random.rand(*W_res.shape) > 0.12] = 0.0 
W_out = np.random.uniform(0.8, 1.8, N_R)

def get_spk(th1, w1, th2, w2):
    st = [th1, w1, th2, w2]
    cur = []
    for i, v in enumerate(st):
        lin = np.linspace(-2.0, 2.0, 16) if i % 2 == 0 else np.linspace(-5.0, 5.0, 16)
        cur.extend(np.exp(-((lin - v) ** 2) / (2 * 0.2 ** 2)))
    return np.array(cur)

def dynamics(th1, w1, th2, w2, t1, t2):
    M11 = (M1 + M2) * L1**2
    M12 = M2 * L1 * L2 * np.cos(th1 - th2)
    M22 = M2 * L2**2
    F1 = -M2 * L1 * L2 * w2**2 * np.sin(th1 - th2) + (M1 + M2) * G * L1 * np.sin(th1) + t1
    F2 =  M2 * L1 * L2 * w1**2 * np.sin(th1 - th2) + M2 * G * L2 * np.sin(th2) + t2
    M = np.array([[M11, M12], [M12, M22]])
    F = np.array([F1, F2])
    return np.linalg.solve(M, F)

h_th1, h_th2, h_snn = np.zeros(N), np.zeros(N), np.zeros(N)
th1, w1, th2, w2 = th1_targ, 0.0, th2_targ, 0.0
int_th1, int_th2 = 0.0, 0.0

for f in range(N):
    h_th1[f], h_th2[f] = th1, th2
    snn_sum = 0.0
    
    for sub in range(S_STEPS):
        int_th1 += th1 * DT_P
        int_th2 += th2 * DT_P
        
        spk = get_spk(th1, w1, th2, w2)
        inj = np.dot(W_in, spk) + np.dot(W_res, (v_res >= 1.0).astype(float))
        v_res = (beta * v_res) + inj
        act = (v_res >= 1.0).astype(float)
        v_res[v_res >= 1.0] = 0.0
        
        f_snn = np.dot(W_out, act)
        snn_sum += f_snn
        
        u1 = -120.0 * th1 - 25.0 * w1 - 45.0 * int_th1
        u2 = -50.0 * th2 - 12.0 * w2 - 20.0 * int_th2
        
        t1 = np.clip(u1 + (f_snn * 0.05), -150.0, 150.0)
        t2 = np.clip(u2 - (f_snn * 0.02), -75.0, 75.0)
        
        k1_dw1, k1_dw2 = dynamics(th1, w1, th2, w2, t1, t2)
        
        th1_2, w1_2, th2_2, w2_2 = th1+0.5*DT_P*w1, w1+0.5*DT_P*k1_dw1, th2+0.5*DT_P*w2, w2+0.5*DT_P*k1_dw2
        k2_dw1, k2_dw2 = dynamics(th1_2, w1_2, th2_2, w2_2, t1, t2)
        
        th1_3, w1_3, th2_3, w2_3 = th1+0.5*DT_P*w1_2, w1+0.5*DT_P*k2_dw1, th2+0.5*DT_P*w2_2, w2+0.5*DT_P*k2_dw2
        k3_dw1, k3_dw2 = dynamics(th1_3, w1_3, th2_3, w2_3, t1, t2)
        
        th1_4, w1_4, th2_4, w2_4 = th1+DT_P*w1_3, w1+DT_P*k3_dw1, th2+DT_P*w2_3, w2+DT_P*k3_dw2
        k4_dw1, k4_dw2 = dynamics(th1_4, w1_4, th2_4, w2_4, t1, t2)
        
        th1 += DT_P * (w1 + 2.0 * w1_2 + 2.0 * w1_3 + w1_4) / 6.0
        w1  += DT_P * (k1_dw1 + 2.0 * k2_dw1 + 2.0 * k3_dw1 + k4_dw1) / 6.0
        th2 += DT_P * (w2 + 2.0 * w2_2 + 2.0 * w2_3 + w2_4) / 6.0
        w2  += DT_P * (k1_dw2 + 2.0 * k2_dw2 + 2.0 * k3_dw2 + k4_dw2) / 6.0

    h_snn[f] = snn_sum / S_STEPS

x1_p, y1_p = L1 * np.sin(h_th1), L1 * np.cos(h_th1)
x2_p, y2_p = x1_p + L2 * np.sin(h_th2), y1_p + L2 * np.cos(h_th2)

fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(3, 2, width_ratios=[1.2, 1.0])

ax_anim = fig.add_subplot(gs[:, 0])
bnd = L1 + L2 + 0.3
ax_anim.set_xlim(-bnd, bnd)
ax_anim.set_ylim(-0.5, bnd + 0.2)
ax_anim.set_aspect('equal')
ax_anim.grid(True, linestyle='--', alpha=0.3)
ax_anim.set_title("Double Pendulum System Space", fontsize=10, fontweight='bold')
ax_anim.plot([0, 0], [-0.5, L1+L2], 'r--', alpha=0.3)
line, = ax_anim.plot([], [], 'o-', lw=4, color='#2ca02c', markersize=10, markerfacecolor='#d62728')

W_WIN = 1.5  
ax_wfp = fig.add_subplot(gs[0, 1])
ax_wfp.grid(True, linestyle='--', alpha=0.3)
ax_wfp.set_title("High-Magnification Scope (1.5s Rolling)", fontsize=10, fontweight='bold')
ax_wfp.set_ylabel("Th 1 (rad)", fontsize=8)
ax_wfp.set_ylim(np.min(h_th1) - 0.2, np.max(h_th1) + 0.2)
tr_p, = ax_wfp.plot([], [], color='#1f77b4', lw=2)

ax_wfy = fig.add_subplot(gs[1, 1])
ax_wfy.grid(True, linestyle='--', alpha=0.3)
ax_wfy.set_ylabel("Th 2 (rad)", fontsize=8)
ax_wfy.set_ylim(np.min(h_th2) - 0.2, np.max(h_th2) + 0.2)
tr_y, = ax_wfy.plot([], [], color='#2ca02c', lw=2)

ax_wfs = fig.add_subplot(gs[2, 1])
ax_wfs.grid(True, linestyle='--', alpha=0.3)
ax_wfs.set_ylabel("SNN (N*m)", fontsize=8)
ax_wfs.set_xlabel("Time (s)", fontsize=8)
ax_wfs.set_ylim(np.min(h_snn) - 1.0, np.max(h_snn) + 1.0)
tr_s, = ax_wfs.plot([], [], color='#e377c2', lw=1.5)

ax_wfp.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfy.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfs.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
plt.tight_layout()

def init():
    line.set_data([], [])
    tr_p.set_data([], [])
    tr_y.set_data([], [])
    tr_s.set_data([], [])
    return line, tr_p, tr_y, tr_s

def animate(i):
    t_c = t_eval[i]
    line.set_data([0, x1_p[i], x2_p[i]], [0, y1_p[i], y2_p[i]])
    
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + DT_F
    t_e = W_WIN if t_c < W_WIN else t_c + DT_F
    ax_wfp.set_xlim(t_s, t_e)
    ax_wfy.set_xlim(t_s, t_e)
    ax_wfs.set_xlim(t_s, t_e)
    
    tr_p.set_data(t_eval[:i], h_th1[:i])
    tr_y.set_data(t_eval[:i], h_th2[:i])
    tr_s.set_data(t_eval[:i], h_snn[:i])
    return line, tr_p, tr_y, tr_s

print("Starting display panels...")
ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=DT_F * 1000, blit=True, repeat=False)
plt.show()