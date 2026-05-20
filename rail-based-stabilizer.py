import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

print("="*40)
print("     CART-POLE SNN ENGINE RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

x_init  = get_in("Cart Pos (m)", 0.1)
dx_init = get_in("Cart Vel (m/s)", 0.0)
th_init = get_in("Pole Ang (rad)", 0.72)
w_init  = get_in("Ang Vel (rad/s)", -0.98)

S0 = [x_init, dx_init, th_init, w_init]
print("-" * 40)
print("Initializing simulation buffers...")

G = 9.81
M1, M2, L = 0.4, 1.0, 1.0       

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

def get_spk(x, dx, th, w):
    st = [x, dx, th, w]
    cur = []
    for i, v_val in enumerate(st):
        lin = np.linspace(-1.0, 1.0, 16)
        c = (lin ** 3) * [4.0, 12.0, 2.0, 18.0][i]
        cur.extend(np.exp(-((c - v_val) ** 2) / (2 * 0.08 ** 2)))
    return np.array(cur)

h_x, h_th, h_snn = np.zeros(N), np.zeros(N), np.zeros(N)
x, dx, th, w = S0
ix = 0.0

for f in range(N):
    h_x[f], h_th[f] = x, th
    snn_sum = 0.0
    
    for sub in range(S_STEPS):
        if np.abs(th) < 0.015 and np.abs(w) < 0.1:
            th, w, dx, ix = 0.0, 0.0, 0.0, 0.0
            v *= 0.5  
            
        ix = (ix * 0.998) + x * DT_P
        ix = np.clip(ix, -2.0, 2.0) 
        
        spk = get_spk(x, dx, th, w)
        inj = np.dot(W_in, spk) + np.dot(W_res, (v >= 0.2).astype(float))
        v = (beta * v) + inj
        act = (v >= 0.2).astype(float)
        v[v >= 0.2] -= 0.2  
        
        K_x, K_dx, K_th, K_w = -5.5, -11.0, 45.0, 11.5
        f_teach = (K_x * x) + (K_dx * dx) + (K_th * th) + (K_w * w) - (5.5 * ix)
        f_teach += np.tanh(th * 8.0) * 100.0 + (14.0 * np.sin(th) * np.abs(dx))
        f_teach = np.clip(f_teach, -450.0, 450.0)
        
        f_snn = np.dot(W_out, act) * 50.0
        snn_sum += f_snn
        
        err = f_snn - f_teach
        k_v = np.dot(P_mat, act)
        den = 1.0 + np.dot(act, k_v)
        if den > 1e-6:
            k_v /= den
            P_mat -= np.outer(k_v, np.dot(act, P_mat))
            W_out -= k_v * (err / 50.0)
            
        f_tot = np.clip(f_snn, -500.0, 500.0) 
        sn, cs = np.sin(th), np.cos(th)
        
        tmp = (f_tot + M2 * L * w**2 * sn) / (M1 + M2)
        d2th = (G * sn - cs * tmp) / (L * (4.0/3.0 - (M2 * cs**2) / (M1 + M2)))
        d2x  = tmp - (M2 * L * d2th * cs) / (M1 + M2)
        
        dx += d2x * DT_P
        x  += dx * DT_P
        w  += d2th * DT_P
        th += w * DT_P
        
    h_snn[f] = snn_sum / S_STEPS

x0, y0 = h_x, np.zeros_like(h_x)
x1, y1 = x0 + L * np.sin(h_th), L * np.cos(h_th)

fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(3, 2, width_ratios=[1.2, 1.0])

ax_anim = fig.add_subplot(gs[:, 0])
ax_anim.set_xlim(-6.0, 6.0)  
ax_anim.set_ylim(-0.3, L + 0.5)
ax_anim.set_aspect('equal')
ax_anim.grid(True, linestyle='--', alpha=0.3)
ax_anim.set_title("Live Mechanical Simulation", fontsize=10, fontweight='bold')
ax_anim.plot([-6.5, 6.5], [0, 0], 'k-', lw=2, alpha=0.6)  
ax_anim.plot([0, 0], [-0.3, L+0.4], 'r--', alpha=0.3)
cart, = ax_anim.plot([], [], 's', markersize=20, color='#7f7f7f')
line, = ax_anim.plot([], [], 'o-', lw=5, color='#2ca02c', markersize=10, markerfacecolor='#d62728')

W_WIN = 1.5  
ax_wfx = fig.add_subplot(gs[0, 1])
ax_wfx.grid(True, linestyle='--', alpha=0.3)
ax_wfx.set_title("SNN Diagnostic Scope (Zoomed 1.5s Window)", fontsize=10, fontweight='bold')
ax_wfx.set_ylabel("Cart Pos (m)", fontsize=8)
ax_wfx.set_ylim(np.min(h_x) - 0.5, np.max(h_x) + 0.5)
tr_x, = ax_wfx.plot([], [], color='#1f77b4', lw=2)

ax_wfth = fig.add_subplot(gs[1, 1])
ax_wfth.grid(True, linestyle='--', alpha=0.3)
ax_wfth.set_ylabel("Pole Ang (rad)", fontsize=8)
ax_wfth.set_ylim(np.min(h_th) - 0.2, np.max(h_th) + 0.2)
tr_th, = ax_wfth.plot([], [], color='#2ca02c', lw=2)

ax_wff = fig.add_subplot(gs[2, 1])
ax_wff.grid(True, linestyle='--', alpha=0.3)
ax_wff.set_ylabel("SNN Force (N)", fontsize=8)
ax_wff.set_xlabel("Time (s)", fontsize=8)
ax_wff.set_ylim(np.min(h_snn) - 20, np.max(h_snn) + 20)
tr_f, = ax_wff.plot([], [], color='#e377c2', lw=1.5)

ax_wfx.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfth.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wff.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
plt.tight_layout()

def init():
    cart.set_data([], [])
    line.set_data([], [])
    tr_x.set_data([], [])
    tr_th.set_data([], [])
    tr_f.set_data([], [])
    return cart, line, tr_x, tr_th, tr_f

def animate(i):
    t_c = t_eval[i]
    cart.set_data([x0[i]], [y0[i]])
    line.set_data([x0[i], x1[i]], [y0[i], y1[i]])
    
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + DT_F
    t_e = W_WIN if t_c < W_WIN else t_c + DT_F
    ax_wfx.set_xlim(t_s, t_e)
    ax_wfth.set_xlim(t_s, t_e)
    ax_wff.set_xlim(t_s, t_e)
    
    tr_x.set_data(t_eval[:i], h_x[:i])
    tr_th.set_data(t_eval[:i], h_th[:i])
    tr_f.set_data(t_eval[:i], h_snn[:i])
    return cart, line, tr_x, tr_th, tr_f

print("Starting display panels...")
ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=DT_F * 1000, blit=True, repeat=False)
plt.show()