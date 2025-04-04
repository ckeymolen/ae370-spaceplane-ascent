import numpy as np
import matplotlib.pyplot as plt

g = 9.81  # gravitational acceleration [m/s^2]

# ===== RK4 integration method =====
def rk4_step(f, t, y, dt, params):
    k1 = f(t, y, params)
    k2 = f(t + dt/2, y + dt/2 * k1, params)
    k3 = f(t + dt/2, y + dt/2 * k2, params)
    k4 = f(t + dt, y + dt * k3, params)
    return y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

# ===== System dynamics with exponential drag =====
def launch_dynamics(t, state, params):
    x, v = state
    m = params["m0"] - params["alpha"] * t
    if m <= 0:
        return np.array([0.0, 0.0])
    rho = params["rho0"] * np.exp(-x / params["H"])
    D = 0.5 * rho * params["Cd"] * params["A"] * v**2 * np.sign(v)
    a = (params["T"] - D - m * g) / m
    return np.array([v, a])

# ===== RK4 integration loop =====
def simulate(params, t_max=120, dt=0.1):
    t_vals = [0]
    y_vals = [np.array([0.0, 0.0])]
    t = 0
    while t < t_max:
        y_next = rk4_step(launch_dynamics, t, y_vals[-1], dt, params)
        t += dt
        if params["m0"] - params["alpha"] * t <= 0:
            break
        y_vals.append(y_next)
        t_vals.append(t)
    return np.array(t_vals), np.array(y_vals)

# Baseline parameters
base_params = {
    "T": 50000,
    "alpha": 20,
    "m0": 1000,
    "Cd": 0.3,
    "A": 1.0,
    "rho0": 1.225,
    "H": 8500
}

# === Figure 1: RK4 Convergence Study ===
def plot_rk4_convergence():
    def rhs(t, y, params):
        return launch_dynamics(t, y, params)
    
    def simulate_altitude(dt, params, t_final=60):
        t, y = 0, np.array([0.0, 0.0])
        while t < t_final:
            y = rk4_step(rhs, t, y, dt, params)
            t += dt
        return y[0]
    
    dts = [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]
    ref = simulate_altitude(0.001, base_params)
    errors = [abs(simulate_altitude(dt, base_params) - ref) for dt in dts]

    plt.figure()
    plt.loglog(dts, errors, '-o')
    plt.xlabel("Time Step Size (s)")
    plt.ylabel("Final Altitude Error (m)")
    plt.title("RK4 Convergence Study")
    plt.grid(True)

# === Figure 2: Max Altitude vs Burn Rate (Baseline) ===
def plot_max_altitude_vs_burn_rate():
    alphas = [10, 20, 30, 40, 50]
    max_alts = []
    for a in alphas:
        p = base_params.copy()
        p["alpha"] = a
        _, y = simulate(p)
        max_alts.append(np.max(y[:, 0]) / 1000)
    plt.figure()
    plt.plot(alphas, max_alts, 'o-')
    plt.xlabel("Burn Rate [kg/s]")
    plt.ylabel("Max Altitude [km]")
    plt.title("Max Altitude vs Burn Rate")
    plt.grid(True)

# === Figure 3: Max Altitude vs Drag Coefficient (Baseline) ===
def plot_max_altitude_vs_drag():
    Cds = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_alts = []
    for c in Cds:
        p = base_params.copy()
        p["Cd"] = c
        _, y = simulate(p)
        max_alts.append(np.max(y[:, 0]) / 1000)
    plt.figure()
    plt.plot(Cds, max_alts, 's-')
    plt.xlabel("Drag Coefficient")
    plt.ylabel("Max Altitude [km]")
    plt.title("Max Altitude vs Drag Coefficient")
    plt.grid(True)

# === Figure 4: Velocity vs Altitude (Ascent Phase) ===
def plot_velocity_vs_altitude_ascent():
    _, y = simulate(base_params)
    x_vals = y[:, 0] / 1000
    v_vals = y[:, 1]
    ascent = v_vals > 0
    plt.figure()
    plt.plot(x_vals[ascent], v_vals[ascent])
    plt.xlabel("Altitude [km]")
    plt.ylabel("Velocity [m/s]")
    plt.title("Velocity vs Altitude (Ascent)")
    plt.grid(True)

# === Figure 5: Kármán Line Feasibility Map ===
def plot_karman_map():
    thrusts = np.linspace(20000, 60000, 8)
    alphas = np.linspace(10, 50, 6)
    T_vals, A_vals = np.meshgrid(thrusts, alphas)
    crossed = np.zeros_like(T_vals, dtype=bool)
    for i in range(T_vals.shape[0]):
        for j in range(T_vals.shape[1]):
            p = base_params.copy()
            p["T"] = T_vals[i, j]
            p["alpha"] = A_vals[i, j]
            _, y = simulate(p)
            crossed[i, j] = np.max(y[:, 0]) > 100000
    plt.figure()
    plt.contourf(T_vals / 1000, A_vals, crossed, levels=[-0.1, 0.5, 1.1], colors=["lightcoral", "lightgreen"])
    plt.xlabel("Thrust [kN]")
    plt.ylabel("Burn Rate [kg/s]")
    plt.title("Thrust vs Burn Rate: Kármán Line Crossing")
    plt.colorbar(ticks=[0, 1], label="Crossed 100 km")
    plt.grid(True)

# === Figure 6: Burn Rate Sweep (m0 = 1500 kg) ===
def alt_vs_burnrate_m1500():
    alphas = [10, 20, 30, 40, 50]
    max_alts = []
    for a in alphas:
        p = base_params.copy()
        p["alpha"] = a
        p["m0"] = 1500
        _, y = simulate(p)
        max_alts.append(np.max(y[:, 0]) / 1000)
    plt.figure()
    plt.plot(alphas, max_alts, 'o-')
    plt.title("Max Altitude vs Burn Rate (m0 = 1500 kg)")
    plt.xlabel("Burn Rate [kg/s]")
    plt.ylabel("Max Altitude [km]")
    plt.grid(True)

# === Figure 7: Drag Sensitivity (T = 30 kN) ===
def alt_vs_drag_T30000():
    Cds = [0.1, 0.2, 0.3, 0.4, 0.5]
    max_alts = []
    for c in Cds:
        p = base_params.copy()
        p["T"] = 30000
        p["Cd"] = c
        _, y = simulate(p)
        max_alts.append(np.max(y[:, 0]) / 1000)
    plt.figure()
    plt.plot(Cds, max_alts, 's-')
    plt.title("Max Altitude vs Drag Coefficient (T = 30 kN)")
    plt.xlabel("Drag Coefficient")
    plt.ylabel("Max Altitude [km]")
    plt.grid(True)

# === Figure 8: Velocity vs Altitude (Heavy Slow Burn) ===
def vel_vs_alt_slowburn_heavy():
    p = base_params.copy()
    p["m0"] = 2000
    p["alpha"] = 10
    _, y = simulate(p)
    x_vals = y[:, 0] / 1000
    v_vals = y[:, 1]
    plt.figure()
    plt.plot(x_vals, v_vals)
    plt.title("Velocity vs Altitude (m0 = 2000 kg, alpha = 10)")
    plt.xlabel("Altitude [km]")
    plt.ylabel("Velocity [m/s]")
    plt.grid(True)

# === Figure 9: Zoomed Kármán Map ===
def karman_map_zoomed():
    thrusts = np.linspace(30000, 50000, 6)
    alphas = np.linspace(10, 40, 7)
    T_vals, A_vals = np.meshgrid(thrusts, alphas)
    crossed = np.zeros_like(T_vals, dtype=bool)
    for i in range(T_vals.shape[0]):
        for j in range(T_vals.shape[1]):
            p = base_params.copy()
            p["T"] = T_vals[i, j]
            p["alpha"] = A_vals[i, j]
            _, y = simulate(p)
            crossed[i, j] = np.max(y[:, 0]) > 100000
    plt.figure()
    plt.contourf(T_vals / 1000, A_vals, crossed, levels=[-0.1, 0.5, 1.1], colors=["lightcoral", "lightgreen"])
    plt.xlabel("Thrust [kN]")
    plt.ylabel("Burn Rate [kg/s]")
    plt.title("Zoomed Thrust vs Burn Rate: Kármán Line")
    plt.colorbar(ticks=[0, 1], label="Crossed 100 km")
    plt.grid(True)

# === Figure 10: Sensitivity Bar Chart (±10% parameter variation) ===
def plot_sensitivity_bar_chart():
    base = base_params.copy()
    base_max_altitude = np.max(simulate(base)[1][:, 0]) / 1000  # km

    param_labels = ['Thrust', 'Burn Rate', 'Drag Coefficient', 'Frontal Area']
    param_keys = ['T', 'alpha', 'Cd', 'A']

    sensitivities = []

    for key in param_keys:
        p_up = base.copy()
        p_down = base.copy()

        p_up[key] *= 1.10
        p_down[key] *= 0.90

        alt_up = np.max(simulate(p_up)[1][:, 0]) / 1000
        alt_down = np.max(simulate(p_down)[1][:, 0]) / 1000

        delta = ((alt_up - alt_down) / (2 * base_max_altitude)) * 100  # percent relative change
        sensitivities.append(delta)

    plt.figure()
    plt.bar(param_labels, sensitivities)
    plt.ylabel("Percent Change in Max Altitude [%]")
    plt.title("Sensitivity to ±10% Parameter Variation")
    plt.grid(axis='y')


# Call all plots
plot_rk4_convergence()
plot_max_altitude_vs_burn_rate()
plot_max_altitude_vs_drag()
plot_velocity_vs_altitude_ascent()
plot_karman_map()
alt_vs_burnrate_m1500()
alt_vs_drag_T30000()
vel_vs_alt_slowburn_heavy()
karman_map_zoomed()
plot_sensitivity_bar_chart()
