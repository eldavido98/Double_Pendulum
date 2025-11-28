# Double-Pendulum

## Overview  
This repository implements a simulation of a **double pendulum** — a classic physics system where one pendulum is attached to the end of another. As with all double-pendulums, the system can exhibit chaotic behavior: tiny differences in initial conditions may lead to dramatically different trajectories. :contentReference[oaicite:0]{index=0}  

This implementation uses JavaScript and HTML to simulate and visualize the motion in a web browser.


## Usage  
1. Clone or download the repository.  
2. Open `double_pendulum.html` in a web browser.  
3. The simulation start automatically

You can interact with the simulation (e.g., adjust initial angles, maybe lengths or masses — depending on what the JS code supports) and visually observe how the pendulum moves and evolves over time.


## What’s inside  
- `double_pendulum.html` — the main HTML file launching the simulation.  
- `double_pendulum.js` — JavaScript logic that computes the pendulum dynamics and handles rendering / animation.  


## What this demonstrates  
- The double pendulum dynamics: two rigid massless rods, point masses at their ends, under gravity (no friction), forming a system governed by coupled non-linear ODEs. :contentReference[oaicite:1]{index=1}  
- Sensitivity to initial conditions: small changes in starting angles, velocities, etc., may lead to radically different motion paths — a hallmark of deterministic chaos. :contentReference[oaicite:2]{index=2}  
- Real-time visualization / animation of the motion, helpful for intuition and teaching/learning dynamics.   


## License & Credits  
Feel free to reuse or adapt this code for educational, research or personal purposes.
