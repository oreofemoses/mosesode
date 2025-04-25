import numpy as np

# Solver descriptions for tooltips
SOLVER_DESCRIPTIONS = {
    "euler": "First-order explicit method. Simple but not very accurate for complex problems.",
    "rk4": "Fourth-order Runge-Kutta method. Good balance of accuracy and performance.",
    "rk45": "Adaptive step size method with error control. Efficient for most problems.",
    "backward_euler": "Implicit first-order method. Stable for stiff ODEs.",
    "crank_nicolson": "Implicit second-order method. Good accuracy and stability.",
    "heun": "Second-order predictor-corrector method. Better accuracy than Euler.",
    "midpoint": "Second-order method. Uses the slope at the midpoint of the interval."
}

# Example ODEs for quick selection
EXAMPLE_ODES = {
    "Linear Decay": {"ode": "-2*y", "y0": "1", "description": "Simple exponential decay: y' = -2y"},
    "Exponential Growth": {"ode": "y", "y0": "1", "description": "Simple exponential growth: y' = y"},
    "Harmonic Oscillator": {"ode": "[y1, -y0]", "y0": "[0, 1]", "description": "Undamped oscillator: y'' + y = 0, written as a system"},
    "Van der Pol": {"ode": "[y1, (1 - y0**2) * y1 - y0]", "y0": "[2, 0]", "description": "Nonlinear oscillator with damping"},
    "Stiff ODE": {"ode": "-50 * (y - np.cos(t))", "y0": "0", "description": "A stiff equation requiring implicit methods for efficiency"}
}

# Comprehensive ODE Library
ODE_LIBRARY = {
    "First-Order ODEs": {
        "Linear Decay": {"ode": "-2*y", "y0": "1", "description": "Simple exponential decay: y' = -2y", "t_end": 3},
        "Exponential Growth": {"ode": "y", "y0": "1", "description": "Simple exponential growth: y' = y", "t_end": 3},
        "Logistic Growth": {"ode": "0.1*y*(1-y/10)", "y0": "1", "description": "Population growth with carrying capacity", "t_end": 50},
        "Sinusoidal Forcing": {"ode": "sin(t)", "y0": "0", "description": "Simple forced system: y' = sin(t)", "t_end": 10},
        "Nonlinear Term": {"ode": "y^2", "y0": "1", "description": "Nonlinear ODE with quadratic term", "t_end": 0.8},
        "Mixed Terms": {"ode": "t^2 + y^2", "y0": "0", "description": "Mix of time and state variables", "t_end": 2},
        "Exponential Decay": {"ode": "exp(-t)*y", "y0": "1", "description": "Time-dependent decay", "t_end": 5},
        "Rational Function": {"ode": "1/(1+t^2)", "y0": "0", "description": "Integration of a rational function", "t_end": 8},
        "Square Root": {"ode": "sqrt(t+1)", "y0": "0", "description": "Involves a square root term", "t_end": 10},
        "Trigonometric": {"ode": "cos(y)", "y0": "0", "description": "Solution depends on its own value through cosine", "t_end": 6}
    },
    "Second-Order ODEs": {
        "Simple Harmonic": {"ode": "D2y = -y", "y0": "[0, 1]", "description": "Undamped harmonic oscillator", "t_end": 10},
        "Free Fall": {"ode": "D2y = -9.81", "y0": "[0, 0]", "description": "Gravitational acceleration", "t_end": 5},
        "Damped Pendulum": {"ode": "D2y = -0.1*D1y - sin(y)", "y0": "[0.5, 0]", "description": "Damped nonlinear pendulum", "t_end": 20},
        "Forced Damped": {"ode": "D2y = sin(t) - 2*D1y - y", "y0": "[0, 0]", "description": "Forced and damped oscillator", "t_end": 20},
        "Nonlinear Spring": {"ode": "D2y = -D1y - y - 0.1*y^3", "y0": "[2, 0]", "description": "Oscillator with nonlinear spring", "t_end": 20},
        "Van der Pol": {"ode": "D2y = (1-y^2)*D1y - y", "y0": "[2, 0]", "description": "Van der Pol oscillator", "t_end": 20},
        "Resonance": {"ode": "D2y = sin(t) - 0.1*D1y - y", "y0": "[0, 0]", "description": "Near-resonance forcing", "t_end": 100},
        "Duffing": {"ode": "D2y = 0.3*cos(t) - 0.2*D1y - y - y^3", "y0": "[1, 0]", "description": "Duffing oscillator with forcing", "t_end": 50},
        "Mathieu": {"ode": "D2y = -2*(1 + 0.2*cos(2*t))*y", "y0": "[1, 0]", "description": "Mathieu equation (parametric oscillator)", "t_end": 30},
        "Rayleigh": {"ode": "D2y = -y + D1y*(1-(D1y)^2)", "y0": "[0, 0.1]", "description": "Rayleigh oscillator", "t_end": 40}
    },
    "Higher-Order ODEs": {
        "Third-Order Linear": {"ode": "D3y = -D2y - D1y - y", "y0": "[1, 0, 0]", "description": "Linear third-order system", "t_end": 15},
        "Third-Order Forced": {"ode": "D3y = sin(t)", "y0": "[0, 0, 0]", "description": "Forced third-order system", "t_end": 15},
        "Fourth-Order Oscillator": {"ode": "D4y = y", "y0": "[1, 0, 0, 0]", "description": "Fourth-order oscillatory system", "t_end": 20},
        "Cubic Nonlinearity": {"ode": "D4y = sin(t) - D3y - 6*D2y - 12*D1y - 8*y", "y0": "[0, 0, 0, 0]", "description": "Complex linear system with forcing", "t_end": 15},
        "Fifth-Order Complex": {"ode": "D5y = exp(-t)*D4y - t^2*D3y + cos(t)*D2y - D1y + y*log(t+1)", "y0": "[0, 0, 0, 0, 0]", "description": "Fifth-order with mixed terms", "t_end": 10},
        "Sixth-Order": {"ode": "D6y = t*exp(-t)*sin(t) - D5y + D4y - D3y + D2y - D1y + y", "y0": "[1, 0, 0, 0, 0, 0]", "description": "Sixth-order linear system", "t_end": 15}
    },
    "Systems of ODEs": {
        "Predator-Prey": {"ode": "[0.5*y0*(1-y0/10) - 0.2*y0*y1, -0.1*y1 + 0.1*y0*y1]", "y0": "[10, 2]", "description": "Lotka-Volterra predator-prey model", "t_end": 100},
        "Lorenz System": {"ode": "[10*(y1-y0), y0*(28-y2)-y1, y0*y1-8/3*y2]", "y0": "[1, 1, 1]", "description": "Chaotic Lorenz attractor", "t_end": 30},
        "Rössler System": {"ode": "[-y1-y2, y0+0.2*y1, 0.2+y2*(y0-5.7)]", "y0": "[1, 1, 1]", "description": "Chaotic Rössler attractor", "t_end": 100},
        "Pendulum System": {"ode": "[y1, -0.1*y1 - sin(y0)]", "y0": "[1, 0]", "description": "Pendulum as first-order system", "t_end": 30},
        "SIR Model": {"ode": "[-0.3*y0*y1, 0.3*y0*y1-0.1*y1, 0.1*y1]", "y0": "[0.99, 0.01, 0]", "description": "SIR epidemic model", "t_end": 60},
        "Brusselator": {"ode": "[1 + y0*y0*y1 - 4*y0, 3*y0 - y0*y0*y1]", "y0": "[1, 1]", "description": "Chemical oscillator model", "t_end": 30},
        "Coupled Oscillators": {"ode": "[y1, y3, -y0 - 0.2*(y0-y2), -y2 - 0.2*(y2-y0)]", "y0": "[1, 0, -1, 0]", "description": "Two coupled oscillators", "t_end": 50},
        "Hamiltonian System": {"ode": "[y1, -y0]", "y0": "[1, 0]", "description": "Energy-conserving oscillator", "t_end": 10}
    },
    "Stiff ODEs": {
        "HIRES": {"ode": "[-1.71*y0 + 0.43*y1 + 8.32*y2 + 0.0007, 1.71*y0 - 8.75*y1, -10.03*y2 + 0.43*y3 + 0.035*y4, 8.32*y1 + 1.71*y2 - 1.12*y3, -1.745*y4 + 0.43*y5 + 0.43*y6, -280*y5*y7 + 0.69*y3 + 1.71*y4 - 0.43*y5 + 0.69*y6, 280*y5*y7 - 1.81*y6, -280*y5*y7 + 1.81*y6]", "y0": "[1, 0, 0, 0, 0, 0, 0, 0.0057]", "description": "High Irradiance RESponse problem", "t_end": 1},
        "Robertson": {"ode": "[-0.04*y0 + 1e4*y1*y2, 0.04*y0 - 1e4*y1*y2 - 3e7*y1*y1, 3e7*y1*y1]", "y0": "[1, 0, 0]", "description": "Chemical kinetics problem", "t_end": 10},
        "Van der Pol (stiff)": {"ode": "[y1, 1000*(1-y0*y0)*y1-y0]", "y0": "[2, 0]", "description": "Stiff Van der Pol oscillator", "t_end": 3000},
        "Linear Stiff": {"ode": "[-100*(y - cos(t))]", "y0": "0", "description": "Simple linear stiff equation", "t_end": 10},
        "Multi-Scale": {"ode": "D4y = -0.001*D3y - 1000*D2y - 0.01*D1y - 100*y + sin(100*t)", "y0": "[0, 0, 0, 0]", "description": "Multi-scale system with fast and slow dynamics", "t_end": 2}
    },
    "Physics Problems": {
        "Planetary Orbit": {"ode": "[y2, y3, -y0/((y0*y0+y1*y1)^1.5), -y1/((y0*y0+y1*y1)^1.5)]", "y0": "[1, 0, 0, 1]", "description": "Two-body gravitational problem", "t_end": 20},
        "RLC Circuit": {"ode": "D2y = 10*sin(20*t) - 5*D1y - 100*y", "y0": "[0, 0]", "description": "Second-order RLC circuit", "t_end": 2},
        "Airy Equation": {"ode": "D2y = t*y", "y0": "[0.5, 0.5]", "description": "Airy's differential equation", "t_end": 15},
        "Heat Equation": {"ode": "D1y = 0.1*(10-y)", "y0": "25", "description": "Simple cooling model", "t_end": 50},
        "Elastic Pendulum": {"ode": "[y2, y4, -2*y2*y4/y0 - (9.81/y0)*sin(y1), y4, -9.81*cos(y1) - 2*y0*y2*y2 + y0*y4*y4]", "y0": "[1, 0.1, 0, 0, 0]", "description": "Elastic pendulum", "t_end": 10}
    }
} 