import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import argparse

# Objective function
def objective(vars):
    x, y = vars
    return (x**2 - 4*x*y + 2*y**2)

# Constraint 1: x + 2y <= 30
def constraint1(vars):
    x, y = vars
    return 30 - (x + 2*y)

# Constraint 2: xy >= 50
def constraint2(vars):
    x, y = vars
    return x*y - 50

# Constraint 3: y <= (3x^2)/100 + 5
def constraint3(vars):
    x, y = vars
    return (3*(x**2))/100 + 5 - y

# Bounds and constraints setup for SLSQP optimizer
def solve_optimization(bounds):
    x_bounds = bounds[0]
    y_bounds = bounds[1]
    
    # Constraints
    cons = ({'type': 'ineq', 'fun': constraint1},
            {'type': 'ineq', 'fun': constraint2},
            {'type': 'ineq', 'fun': constraint3})
    
    # Initial guess
    initial_guess = [1, 1]

    # Solving the optimization problem
    solution = minimize(objective, initial_guess, method='SLSQP', bounds=[x_bounds, y_bounds], constraints=cons)
    
    return solution

# Plotting the feasible region and the optimal point
def plot_feasible_region(x_opt, y_opt):
    x = np.linspace(0.1, 40, 400)  # Avoid x=0 to prevent division by zero in xy >= 50
    
    # Compute constraints for the 1D array `x`
    y1 = (30 - x) / 2  # x + 2y <= 30
    y2 = 50 / x        # xy >= 50 (hyperbolic)
    y3 = (3 * x**2) / 100 + 5  # y <= 3x^2/100 + 5 (parabola)
    
    # Ensure we only plot within reasonable bounds
    y1 = np.clip(y1, 0, 33)
    y2 = np.clip(y2, 0, 33)
    y3 = np.clip(y3, 0, 15)

    plt.figure(figsize=(10, 8))

    # Plot constraints as lines
    plt.plot(x, y1, label='x + 2y ≤ 30', color='red')
    plt.plot(x, y2, label='xy ≥ 50', color='blue')
    plt.plot(x, y3, label='y ≤ 3x^2/100 + 5', color='green')

    # Fill the feasible region between the constraints
    plt.fill_between(x, np.maximum(y2, 0), np.minimum(y1, y3), where=(y1 >= y2) & (y3 >= y2), color='grey', alpha=0.5)

    # Plot the optimal point
    plt.plot(x_opt, y_opt, 'ro', label=f'Optimal Point ({x_opt:.2f}, {y_opt:.2f})')
    plt.text(x_opt, y_opt, f'({x_opt:.2f}, {y_opt:.2f})', color='black', fontsize=12, ha='right')

    # Plot the objective function
    X, Y = np.meshgrid(np.linspace(0, 33, 100), np.linspace(0, 15, 100))  
    Z = -X**2 + 4*X*Y - 2*Y**2 

    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
    plt.clabel(contour, inline=True, fontsize=8)

    # Set axis limits and labels
    plt.xlim([0, 30])
    plt.ylim([0, 12])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Feasible Region and Constraints with Optimal Point')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function to handle input arguments and execute the optimization
def main():
    parser = argparse.ArgumentParser(description='Maximize the profit function subject to constraints.')
    parser.add_argument('--xmax', type=float, default=15, help='Upper bound for x')
    parser.add_argument('--ymax', type=float, default=15, help='Upper bound for y')
    
    args = parser.parse_args()

    # Define bounds with 0 for default lower value
    x_bounds = (0, args.xmax)
    y_bounds = (0, args.ymax)
    
    # Solve the optimization problem
    solution = solve_optimization([x_bounds, y_bounds])
    x_opt, y_opt = solution.x
    
    print(f"Optimal values:\nx = {x_opt:.2f}, y = {y_opt:.2f}")
    print(f"Maximum Profit: {-solution.fun:.2f}")
    
    # Plot the feasible region and the optimal point
    plot_feasible_region(x_opt, y_opt)

if __name__ == '__main__':
    main()
