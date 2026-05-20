import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

print("="*40)
print("     TRMS SNN ENGINE RUNTIME")
print("="*40)

def get_in(msg, val):
    inp = input(f"{msg} [{val}]: ").strip()
    return float(val) if inp == "" else float(inp)

p_targ = get_in("Pitch (rad)", 0.35)
y_targ = get_in("Yaw (rad)", -0.50)

print("-" * 40)
print("Initializing simulation buffers...")

G = 9.81     
I_P, I_Y = 0.015, 0.022    
B_P, B_Y = 0.012, 0.018    
M_A, R_A = 0.35, 0.25     

K_ML, K_TY = 0.12, 0.08  
K_CY, K_GP = 0.03, 0.015 

T_MAX = 16.0
FPS = 60
DT_F = 1.0 / FPS
t_eval = np.arange(0, T_MAX, DT_F)
N = len(t_eval)
S_STEPS = 25  
DT_P = DT_F / S_STEPS  

N_S, N_R = 68, 300
np.random.seed(42)

beta = np.random.uniform(0.94, 0.99, N_R)
v_res = np.zeros(N_R)
W_in = np.random.uniform(-0.5, 0.5, (N_R, N_S))
W_res = np.random.randn(N_R, N_R) * 0.05
W_res[np.random.rand(*W_res.shape) > 0.2] = 0.0
W_out = np.zeros(N_R)
P_mat = np.eye(N_R) * 0.1  

def get_spk(th, dth, ps, dps):
    st = [th, dth, ps, dps]
    cur = []
    for i, v in enumerate(st):
        lin = np.linspace(-1.0, 1.0, 17)
        c = (lin ** 3) * [2.0, 8.0, 3.14, 8.0][i]
        cur.extend(np.exp(-((c - v) ** 2) / (2 * 0.08 ** 2)))
    return np.array(cur)

def dynamics(th, dth, ps, dps, V_m, V_t):
    t_g = M_A * G * R_A * np.cos(th)
    t_p = (K_ML * V_m) - (B_P * dth) - t_g - (K_GP * dps * np.abs(V_m))
    t_y = (K_TY * V_t) - (B_Y * dps) + (K_CY * V_m)
    return t_p / I_P, t_y / I_Y

h_th, h_ps, h_snn = np.zeros(N), np.zeros(N), np.zeros(N)
th, dth, ps, dps = 0.0, 0.0, 0.0, 0.0
ie_p, ie_y = 0.0, 0.0

for f in range(N):
    h_th[f], h_ps[f] = th, ps
    snn_sum = 0.0
    
    for sub in range(S_STEPS):
        e_p, e_y = p_targ - th, y_targ - ps
        ie_p = np.clip(ie_p + e_p * DT_P, -2.0, 2.0)
        ie_y = np.clip(ie_y + e_y * DT_P, -2.0, 2.0)
        
        spk = get_spk(th, dth, ps, dps)
        inj = np.dot(W_in, spk) + np.dot(W_res, (v_res >= 0.2).astype(float))
        v_res = (beta * v_res) + inj
        act = (v_res >= 0.2).astype(float)
        v_res[v_res >= 0.2] -= 0.2  
        
        V_mb = 35.0 * e_p - 8.0 * dth + 15.0 * ie_p + 11.5 
        V_tb = 22.0 * e_y - 6.0 * dps + 8.0 * ie_y
        
        f_snn = np.dot(W_out, act) * 20.0
        snn_sum += f_snn
        
        V_m = np.clip(V_mb, -100.0, 100.0)
        V_t = np.clip(V_tb - f_snn, -60.0, 60.0)
        
        t_targ = (K_CY * V_m) / K_TY
        err = f_snn - t_targ
        k_v = np.dot(P_mat, act)
        den = 1.0 + np.dot(act, k_v)
        if den > 1e-5:
            k_v /= den
            P_mat -= np.outer(k_v, np.dot(act, P_mat))
            W_out -= k_v * (err / 20.0)
            
        k1_dp, k1_dy = dynamics(th, dth, ps, dps, V_m, V_t)
        
        th2, dth2, ps2, dps2 = th+0.5*DT_P*dth, dth+0.5*DT_P*k1_dp, ps+0.5*DT_P*dps, dps+0.5*DT_P*k1_dy
        k2_dp, k2_dy = dynamics(th2, dth2, ps2, dps2, V_m, V_t)
        
        th3, dth3, ps3, dps3 = th+0.5*DT_P*dth2, dth+0.5*DT_P*k2_dp, ps+0.5*DT_P*dps2, dps+0.5*DT_P*k2_dy
        k3_dp, k3_dy = dynamics(th3, dth3, ps3, dps3, V_m, V_t)
        
        th4, dth4, ps4, dps4 = th+DT_P*dth3, dth+DT_P*k3_dp, ps+DT_P*dps3, dps+DT_P*k3_dy
        k4_dp, k4_dy = dynamics(th4, dth4, ps4, dps4, V_m, V_t)
        
        th  += DT_P * (dth + 2.0 * dth2 + 2.0 * dth3 + dth4) / 6.0
        dth += DT_P * (k1_dp + 2.0 * k2_dp + 2.0 * k3_dp + k4_dp) / 6.0
        ps  += DT_P * (dps + 2.0 * dps2 + 2.0 * dps3 + dps4) / 6.0
        dps += DT_P * (k1_dy + 2.0 * k2_dy + 2.0 * k3_dy + k4_dy) / 6.0

    h_snn[f] = snn_sum / S_STEPS

L_B, Z_P = 1.2, 1.0
x_m3d = L_B * np.sin(h_ps) * np.cos(h_th)
y_m3d = L_B * np.cos(h_ps) * np.cos(h_th)
z_m3d = Z_P + L_B * np.sin(h_th)
x_t3d = -L_B * np.sin(h_ps) * np.cos(h_th)
y_t3d = -L_B * np.cos(h_ps) * np.cos(h_th)
z_t3d = Z_P - L_B * np.sin(h_th)

xr3d = L_B * np.sin(y_targ) * np.cos(p_targ)
yr3d = L_B * np.cos(y_targ) * np.cos(p_targ)
zr3d = Z_P + L_B * np.sin(p_targ)

x_p_rod, y_p_rod = L_B * np.cos(h_th), L_B * np.sin(h_th)
xp_t, yp_t = L_B * np.cos(p_targ), L_B * np.sin(p_targ)
x_y_rod, y_y_rod = L_B * np.sin(h_ps), L_B * np.cos(h_ps)
xy_t, yy_t = L_B * np.sin(y_targ), L_B * np.cos(y_targ)

fig = plt.figure(figsize=(16, 8.5))
gs = GridSpec(3, 3, figure=fig, width_ratios=[1.0, 1.0, 1.2])

ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_3d.set_xlim(-1.4, 1.4)
ax_3d.set_ylim(-1.4, 1.4)
ax_3d.set_zlim(0.0, 2.2)
ax_3d.set_title("3D Spatial View", fontsize=10, fontweight='bold')
ax_3d.plot([0, 0], [0, 0], [0, Z_P], color='#404040', lw=5)
b_3d, = ax_3d.plot([], [], [], 'o-', color='#7f7f7f', lw=3, markersize=3)
m_3d, = ax_3d.plot([], [], [], 'o', color='#d62728', markersize=12)
t_3d, = ax_3d.plot([], [], [], 'o', color='#1f77b4', markersize=8)
ax_3d.plot([xr3d], [yr3d], [zr3d], 'rX', markersize=10, alpha=0.7)
ax_3d.view_init(elev=20, azim=-45)

ax_p = fig.add_subplot(gs[:2, 1])
ax_p.set_xlim(-1.4, 1.4)
ax_p.set_ylim(-1.4, 1.4)
ax_p.set_aspect('equal')
ax_p.grid(True, linestyle='--', alpha=0.3)
ax_p.set_title("2D Side View: PITCH", fontsize=10, fontweight='bold')
ax_p.plot([0, 0], [-1.4, 1.4], 'k--', alpha=0.15)
ax_p.plot([-1.4, 1.4], [0, 0], 'k-', alpha=0.3)
lp, = ax_p.plot([], [], 'o-', lw=4, color='#d62728', markersize=7)
ax_p.plot([xp_t], [yp_t], 'rX', markersize=9, alpha=0.7)

ax_y = fig.add_subplot(gs[2, 1])
ax_y.set_xlim(-1.4, 1.4)
ax_y.set_ylim(-1.4, 1.4)
ax_y.set_aspect('equal')
ax_y.grid(True, linestyle='--', alpha=0.3)
ax_y.set_title("2D Top View: YAW", fontsize=10, fontweight='bold')
ax_y.plot([0, 0], [-1.4, 1.4], 'k-', alpha=0.3)
ax_y.plot([-1.4, 1.4], [0, 0], 'k--', alpha=0.15)
ly, = ax_y.plot([], [], 'o-', lw=4, color='#1f77b4', markersize=7)
ax_y.plot([xy_t], [yy_t], 'rX', markersize=9, alpha=0.7)

W_WIN = 1.5  
ax_wfp = fig.add_subplot(gs[0, 2])
ax_wfp.grid(True, linestyle='--', alpha=0.3)
ax_wfp.set_title("SNN Diagnostic Scope (Zoomed 1.5s Window)", fontsize=10, fontweight='bold')
ax_wfp.set_ylabel("Pitch (rad)", fontsize=8)
ax_wfp.set_ylim(-0.2, np.max(h_th) + 0.2)
ax_wfp.axhline(p_targ, color='r', linestyle='--', alpha=0.5)
tr_p, = ax_wfp.plot([], [], color='#d62728', lw=2)

ax_wfy = fig.add_subplot(gs[1, 2])
ax_wfy.grid(True, linestyle='--', alpha=0.3)
ax_wfy.set_ylabel("Yaw (rad)", fontsize=8)
ax_wfy.set_ylim(np.min(h_ps) - 0.2, 0.2)
ax_wfy.axhline(y_targ, color='r', linestyle='--', alpha=0.5)
tr_y, = ax_wfy.plot([], [], color='#1f77b4', lw=2)

ax_wfs = fig.add_subplot(gs[2, 2])
ax_wfs.grid(True, linestyle='--', alpha=0.3)
ax_wfs.set_ylabel("SNN (V)", fontsize=8)
ax_wfs.set_xlabel("Time (s)", fontsize=8)
ax_wfs.set_ylim(np.min(h_snn) - 2.0, np.max(h_snn) + 2.0)
tr_s, = ax_wfs.plot([], [], color='#e377c2', lw=1.5)

ax_wfp.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfy.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
ax_wfs.xaxis.set_major_locator(ticker.MultipleLocator(0.25))
plt.tight_layout()

import matplotlib.lines as mlines
if not hasattr(mlines.Line2D, 'set_data_new'):
    mlines.Line2D.set_data_new = mlines.Line2D.set_data

def init():
    b_3d.set_data_new([], [])
    b_3d.set_3d_properties([])
    m_3d.set_data_new([], [])
    m_3d.set_3d_properties([])
    t_3d.set_data_new([], [])
    t_3d.set_3d_properties([])
    lp.set_data([], [])
    ly.set_data([], [])
    tr_p.set_data([], [])
    tr_y.set_data([], [])
    tr_s.set_data([], [])
    return b_3d, m_3d, t_3d, lp, ly, tr_p, tr_y, tr_s

def animate(i):
    t_c = t_eval[i]
    b_3d.set_data_new([x_t3d[i], x_m3d[i]], [y_t3d[i], y_m3d[i]])
    b_3d.set_3d_properties([z_t3d[i], z_m3d[i]])
    m_3d.set_data_new([x_m3d[i]], [y_m3d[i]])
    m_3d.set_3d_properties([z_m3d[i]])
    t_3d.set_data_new([x_t3d[i]], [y_t3d[i]])
    t_3d.set_3d_properties([z_t3d[i]])
    
    lp.set_data([0, x_p_rod[i]], [0, y_p_rod[i]])
    ly.set_data([0, x_y_rod[i]], [0, y_y_rod[i]])
    
    t_s = 0.0 if t_c < W_WIN else t_c - W_WIN + DT_F
    t_e = W_WIN if t_c < W_WIN else t_c + DT_F
    ax_wfp.set_xlim(t_s, t_e)
    ax_wfy.set_xlim(t_s, t_e)
    ax_wfs.set_xlim(t_s, t_e)
    
    tr_p.set_data(t_eval[:i], h_th[:i])
    tr_y.set_data(t_eval[:i], h_ps[:i])
    tr_s.set_data(t_eval[:i], h_snn[:i])
    return b_3d, m_3d, t_3d, lp, ly, tr_p, tr_y, tr_s

print("Starting display panels...")
ani = animation.FuncAnimation(fig, animate, frames=N, init_func=init, interval=DT_F * 1000, blit=False, repeat=False)
plt.show()