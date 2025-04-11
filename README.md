# AE370 Project 1 – Spaceplane Vertical Ascent Simulation

This repository contains the full implementation and analysis for **AE370 Project 1**, which investigates the vertical ascent dynamics of a reusable spaceplane. The goal is to evaluate how thrust, drag, fuel burn rate, and mass influence a spaceplane's ability to reach the Kármán line (100 km altitude).

---

##  Project Overview

This project simulates a one-dimensional launch trajectory under:
- Nonlinear atmospheric drag (exponential air density model)
- Constant vertical thrust
- Time-varying mass due to fuel burn
- Gravitational acceleration

We solve the resulting coupled nonlinear ordinary differential equations (ODEs) using a **fourth-order Runge–Kutta (RK4)** integrator. The model supports parameter sweeps, convergence tests, and sensitivity analysis.

---

##  Key Design Questions Answered

| Design Question | Figure(s) |
|-----------------|-----------|
| How do thrust and burn rate affect altitude? | `max_altitude_vs_burnrate_base.png`, `karman_map_base.png` |
| How does drag affect trajectory? | `altitude_vs_drag_base.png`, `altitude_vs_drag_T30k.png` |
| What thrust/burn rate combinations reach 100 km? | `karman_map_base.png`, `karman_map_zoomed.png` |
| What parameters most affect altitude? | `sensitivity_bar_chart.png` |

---

##  Folder Structure
AE370_Spaceplane_Dynamics/ │ ├── code/ # All Python scripts and RK4 integrator ├── results/ # PNG plots generated by simulation └── README.md # This file


---

##  Features

- RK4 implementation with convergence validation  
- Burn rate vs. altitude tradeoff analysis  
- Drag sensitivity plots  
- Kármán line feasibility maps  
- Sensitivity bar chart (+10%/-10% parameter variation)


---
Below are sample figures generated from the simulation and analysis:

### Numerical Verification

**RK4 Convergence Study**  
Validates fourth-order convergence of the Runge-Kutta integration method.  
![RK4 Convergence](figs/RK4%20Convergence%20Study.png)

**RK4 Stability Region**  
Shows the stability region in the complex plane.  
![RK4 Stability](figs/RK4%20Stability%20Region.png)

---

### Baseline Parameter Effects

**Max Altitude vs Burn Rate**  
Explores how increasing fuel consumption rate affects altitude.  
![Burn Rate Sweep](figs/Max%20Altitude%20vs%20Burn%20Rate.png)

**Max Altitude vs Drag Coefficient**  
Drag heavily limits altitude, especially at low thrust.  
![Drag Sensitivity](figs/Max%20Altitude%20vs%20Drag%20Coefficient.png)

**Velocity vs Altitude (Ascent)**  
Displays velocity buildup during vertical ascent.  
![Velocity Ascent](figs/Velocity%20vs%20Altitude%20(Ascent).png)

---

### Parametric Variation Studies

**Burn Rate vs Altitude (m₀ = 1500 kg)**  
Heavier vehicle performance across burn rates.  
![Heavy Burn Rate](figs/Max%20Altitude%20vs%20Burn%20Rate%20(m_0%20=%201500kg).png)

**Drag Sensitivity (T = 30 kN)**  
Altitude reduction with increasing drag under low thrust.  
![Drag T30kN](figs/Max%20Altitude%20vs%20Drag%20Coefficient%20(T=30kN).png)

**Velocity vs Altitude (m₀ = 2000 kg, α = 10)**  
Slower burn and heavy mass generate smoother trajectory.  
![Velocity Heavy](figs/Velocity%20vs%20Altitude%20(m_0%20=%202000kg,%20alpha%20=10).png)

---

### Design Analysis

**Kármán Line Feasibility Map**  
Identifies thrust and burn rate combinations that cross 100 km.  
![Karman Map](figs/Thrust%20vs%20Burn%20Rate%20(Karman%20Line%20Crossing).png)

**Zoomed Kármán Map**  
Finer resolution shows how aggressive burn strategies need more thrust.  
![Zoomed Karman Map](figs/Zoomed%20Thrust%20vs%20Burn%20Rate%20(Karman%20Line).png)

**Sensitivity to ±10% Parameter Variation**  
Bar chart showing which variables impact altitude most.  
![Sensitivity](figs/Sensitivity%20to%20+-%2010%%20Parameter%20Variation.png)

##  How to Run the Simulation

1. Clone the repository:
   ```bash
   git clone https://github.com/ckeymolen/AE370_Spaceplane_Dynamics.git
   cd AE370_Spaceplane_Dynamics

