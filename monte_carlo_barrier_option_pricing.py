import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

np.random.seed(123) # Set seed for reproducibility

# Part 1
# Price the Option
# Price the Barrier Option using Milstein Scheme
def milstein_barrier_option_price(S0, K, B, r, sigma, T, N, M):
    dt = T / N  # Time step size
    discount_factor = np.exp(-r * T)  # Discount factor for final payoff
    
    # Start timing
    start_time = time.perf_counter()

    # Initialise stock prices and barrier condition
    S = np.full(M, S0, dtype=np.float64) # Vector of length M where each element represents stock price of given MC path
    # At each time step i from 1 to N each path updaters its stock price according to Milstein scheme
    barrier_hit = np.zeros(M, dtype=bool)
    
    for _ in range(N):
        dW = np.sqrt(dt) * np.random.randn(M)  # Brownian increments
        S = S + r * S * dt + sigma * S * dW + 0.5 * sigma**2 * S * (dW**2 - dt)
        barrier_hit = barrier_hit | (S < B)  # Mark paths that hit the barrier
    
    # Compute final payoffs, setting zero for paths that hit the barrier
    payoffs = np.maximum(S - K, 0)
    payoffs[barrier_hit] = 0
    
    # Compute Monte Carlo estimates
    aM = discount_factor * np.mean(payoffs)  # Option price estimate
    bM = discount_factor * np.std(payoffs, ddof=1)  # Standard deviation estimate
    
    # Compute actual confidence interval bounds
    ci_lower = aM - 1.96 * (bM / np.sqrt(M))
    ci_upper = aM + 1.96 * (bM / np.sqrt(M))
    confidence_interval = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

    # Compute required M for CI width < $0.01
    required_M = int(np.ceil(((2 * 1.96 * bM) / 0.01) ** 2))  # Solve for M in CI equation - use ceiling to ensure integer value required for MC

    # End timing
    end_time = time.perf_counter()
    comp_time = end_time - start_time # Computational time for M = 10,000

    # Estimate time for required_M
    estimated_time_required_M = (required_M / M) * comp_time 

    return aM, bM, confidence_interval, required_M, comp_time, estimated_time_required_M

# Outputs with our specific parameters 
aM_1, bM_1, confidence_interval_1, required_M_1, comp_time_1, base_time = milstein_barrier_option_price(100, 90, 61, 0.01, 0.1, 1.5, 1000, 10000)

labels = [
    r"$a_M$ (\$)",  # Use raw string (r"...") to avoid double escaping
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)",  # Use {,} for a proper comma in LaTeX
    r"$M^{*}$",
    r"$t^{*}$ (seconds)"
]

values = [
    f"{aM_1:.4f}",
    f"{bM_1:.4f}",
    confidence_interval_1,
    f"{comp_time_1:.4f}",
    f"{required_M_1}",
    f"{base_time:.2f}"
]

# Create DataFrame with labels as column headers and values as a single row
df_results1 = pd.DataFrame([values], columns=labels)

# Save to LaTeX format
df_results1.to_latex("results_part1.tex", index=False, escape=False, column_format="c" * df_results1.shape[1])

# Print the DataFrame
print(df_results1)

# Part 2
# Variance Reduction: Antithetic Variates 
def milstein_barrier_option_price_antithetic(S0, K, B, r, sigma, T, N, M):
    dt = T / N  # Time step size
    discount_factor = np.exp(-r * T)  # Discount factor for final payoff
    
    # Start timing
    start_time = time.perf_counter()
    
    # Generate standard normal random numbers for M paths
    Z = np.random.randn(M, N)  # Standard normal random numbers
    Z_antithetic = -Z  # Antithetic variates
    
    # Initialize stock prices and barrier conditions for both sets
    S = np.full((M, ), S0, dtype=np.float64)
    S_antithetic = np.full((M, ), S0, dtype=np.float64)
    
    barrier_hit = np.zeros(M, dtype=bool)
    barrier_hit_antithetic = np.zeros(M, dtype=bool)
    
    # Simulate paths with Milstein scheme
    for i in range(N):
        dW = np.sqrt(dt) * Z[:, i]
        dW_antithetic = np.sqrt(dt) * Z_antithetic[:, i]
        
        S = S + r * S * dt + sigma * S * dW + 0.5 * sigma**2 * S * (dW**2 - dt)
        S_antithetic = S_antithetic + r * S_antithetic * dt + sigma * S_antithetic * dW_antithetic + 0.5 * sigma**2 * S_antithetic * (dW_antithetic**2 - dt)
        
        barrier_hit |= (S < B)
        barrier_hit_antithetic |= (S_antithetic < B)
    
    # Compute payoffs and apply barrier condition
    payoffs = np.maximum(S - K, 0)
    payoffs_antithetic = np.maximum(S_antithetic - K, 0)
    
    payoffs[barrier_hit] = 0
    payoffs_antithetic[barrier_hit_antithetic] = 0
    
    # Use antithetic variates by averaging both paths
    final_payoffs = (payoffs + payoffs_antithetic) / 2

    # Compute correlation between final stock prices and option payoffs
    correlation_stock = np.corrcoef(S, S_antithetic)[0, 1]
    correlation_payoff = np.corrcoef(payoffs, payoffs_antithetic)[0, 1]
    
    # Monte Carlo estimates
    aM = discount_factor * np.mean(final_payoffs)  # Option price estimate
    bM = discount_factor * np.std(final_payoffs, ddof=1)  # Standard deviation estimate
    
    # Confidence interval bounds
    ci_lower = aM - 1.96 * (bM / np.sqrt(M))
    ci_upper = aM + 1.96 * (bM / np.sqrt(M))
    confidence_interval = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    
    # Compute required M for CI width < $0.01
    required_M = int(np.ceil(((2 * 1.96 * bM) / 0.01) ** 2))
    
    # End timing
    end_time = time.perf_counter()
    comp_time = end_time - start_time  # Computational time for M = 10,000
    
    # Estimate time for required_M
    estimated_time_required_M = (required_M / M) * comp_time

    # Compute speedup factor
    speedup_factor = base_time / estimated_time_required_M
    
    return aM, bM, confidence_interval, required_M, comp_time, estimated_time_required_M, speedup_factor, correlation_stock, correlation_payoff

# Run function with given parameters
aM_2, bM_2, confidence_interval_2, required_M_2, comp_time_2, estimated_time_required_M_2, speedup_factor_2, correlation_stock, correlation_payoff = milstein_barrier_option_price_antithetic(100, 90, 61, 0.01, 0.1, 1.5, 1000, 10000)

labels = [
    r"$a_M$ (\$)",
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)",
    r"$M^{*}$",
    r"$t^{*}$ (seconds)",
    "Speedup Factor"
]

values = [
    f"{aM_2:.4f}",
    f"{bM_2:.4f}",
    confidence_interval_2,
    f"{comp_time_2:.4f}",
    f"{required_M_2}",
    f"{estimated_time_required_M_2:.2f}",
    f"${speedup_factor_2:.4f} \\times$"
]

# Create DataFrame for results
df_results2 = pd.DataFrame([values], columns=labels)

# Create a separate DataFrame for correlation values
df_correlation = pd.DataFrame([[correlation_stock, correlation_payoff]], columns=["Final Stock Price Correlation", "Final Option Price Correlation"])

# Save results to LaTeX format
df_results2.to_latex("results_part2.tex", index=False, escape=False, column_format="c" * df_results2.shape[1])
df_correlation.to_latex("results_correlation.tex", index=False, escape=False, column_format="c" * df_correlation.shape[1])

# Print the DataFrame
print(df_results2)
print(df_correlation)

# Part 3
# Variance Reduction: Fine-Tuned Control Variates
def milstein_barrier_option_control_variate(S0, K, B, r, sigma, T, N, M):
    dt = T / N
    discount_factor = np.exp(-r * T)

    start_time = time.perf_counter()

    # Monte Carlo simulation
    S = np.full(M, S0, dtype=np.float64)
    barrier_hit = np.zeros(M, dtype=bool)

    for _ in range(N):
        dW = np.sqrt(dt) * np.random.randn(M)
        S += r * S * dt + sigma * S * dW + 0.5 * sigma**2 * S * (dW**2 - dt)
        barrier_hit |= (S < B)

    final_prices = S
    payoffs = np.maximum(final_prices - K, 0)
    payoffs[barrier_hit] = 0

    # Estimate theta_opt from the same data
    expected_S = S0 * np.exp(r * T)
    cov_VS = np.cov(payoffs, final_prices, ddof=1)[0, 1]
    var_S = np.var(final_prices, ddof=1)
    var_V = np.var(payoffs, ddof=1)  # Variance of payoffs
    theta_opt = cov_VS / var_S if var_S > 0 else 0

    # Apply control variate adjustment
    adjusted_payoffs = payoffs + theta_opt * (expected_S - final_prices)

    aM = discount_factor * np.mean(adjusted_payoffs)
    bM = discount_factor * np.std(adjusted_payoffs, ddof=1)
    ci_lower = aM - 1.96 * (bM / np.sqrt(M))
    ci_upper = aM + 1.96 * (bM / np.sqrt(M))
    confidence_interval = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

    required_M = int(np.ceil(((2 * 1.96 * bM) / 0.01) ** 2))

    end_time = time.perf_counter()
    comp_time = end_time - start_time
    estimated_time_required_M = (required_M / M) * comp_time
    speedup_factor = base_time / estimated_time_required_M

    return aM, bM, confidence_interval, required_M, comp_time, estimated_time_required_M, speedup_factor, theta_opt, var_S, cov_VS, var_V

# Run function with given parameters
aM_3, bM_3, confidence_interval_3, required_M_3, comp_time_3, estimated_time_required_M_3, speedup_factor_3, theta_opt_3, var_S_3, cov_VS_3, var_V_3  = milstein_barrier_option_control_variate(100, 90, 61, 0.01, 0.1, 1.5, 1000, 10000)

labels = [
    r"$a_M$ (\$)",
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)",
    r"$M^{*}$",
    r"$t^{*}$ (seconds)",
    "Speedup Factor"
]

values = [
    f"{aM_3:.4f}",
    f"{bM_3:.4f}",
    confidence_interval_3,
    f"{comp_time_3:.4f}",
    f"{required_M_3}",
    f"{estimated_time_required_M_3:.2f}",
    f"${speedup_factor_3:.4f} \\times$"
]

# Create DataFrame for results
df_results3 = pd.DataFrame([values], columns=labels)

# Save results to LaTeX format
df_results3.to_latex("results_part3.tex", index=False, escape=False, column_format="c" * df_results3.shape[1])

# Print the DataFrame
print(df_results3)

# Define function for variance of Z_theta
def variance_Z_theta(theta_values, var_V, var_S, cov_VS):
    return var_V - 2 * theta_values * cov_VS + theta_values**2 * var_S

# Generate theta values around optimal theta
theta_values = np.linspace(theta_opt_3 - 1, theta_opt_3 + 1, 100)
var_Z_values = variance_Z_theta(theta_values, var_V_3, var_S_3, cov_VS_3)

# Get the minimum variance of Z_theta at theta_opt
min_var_Z_theta = variance_Z_theta(theta_opt_3, var_V_3, var_S_3, cov_VS_3)

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(theta_values, var_Z_values, 
         label=rf'$\operatorname{{Var}}(Y_{{\theta}}) = {min_var_Z_theta:.4f} \quad \text{{at}} \quad \theta_\text{{opt}}$', 
         color='blue')
plt.axvline(x=theta_opt_3, color='red', linestyle='--', label=rf'$\theta_{{\text{{opt}}}} = {theta_opt_3:.4f}$')
plt.axhline(y=var_V_3, color='green', linestyle='--', label=rf'$\operatorname{{Var}}(h(S_{{N,m}})) = {var_V_3:.4f}$')
plt.xlabel(r'$\theta$', fontsize=12)
plt.ylabel(r'$\operatorname{Var}(Y_{\theta})$', fontsize=12)
plt.title('Variance Reduction with Fine-tuned Control Variates', fontsize=14)
plt.legend(fontsize=10, loc='upper right', frameon=True)
plt.grid()
# Save the plot as a PDF and PNG for use in LaTeX
plt.savefig("variance_reduction_plot.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

# Part 4
# Variance Reduction: Antithetic and Fine-Tuned Control Variates
def milstein_barrier_option_control_antithetic(S0, K, B, r, sigma, T, N, M):
    dt = T / N  # Time step size
    discount_factor = np.exp(-r * T)  # Discount factor for final payoff

    # Start timing
    start_time = time.perf_counter()

    # Step 1: Estimate Cov(V, S) and Var(S) within the main Monte Carlo simulation
    Z = np.random.randn(M, N)  # Generate normal random variables
    Z_antithetic = -Z  # Antithetic variates

    S = np.full(M, S0, dtype=np.float64)
    S_antithetic = np.full(M, S0, dtype=np.float64)

    barrier_hit = np.zeros(M, dtype=bool)
    barrier_hit_antithetic = np.zeros(M, dtype=bool)

    for i in range(N):
        dW = np.sqrt(dt) * Z[:, i]
        dW_antithetic = np.sqrt(dt) * Z_antithetic[:, i]

        S = S + r * S * dt + sigma * S * dW + 0.5 * sigma**2 * S * (dW**2 - dt)
        S_antithetic = S_antithetic + r * S_antithetic * dt + sigma * S_antithetic * dW_antithetic + 0.5 * sigma**2 * S_antithetic * (dW_antithetic**2 - dt)

        barrier_hit |= (S < B)
        barrier_hit_antithetic |= (S_antithetic < B)

    # Compute payoffs and required statistics for control variate estimation
    final_prices = S
    final_prices_antithetic = S_antithetic

    payoffs = np.maximum(final_prices - K, 0)
    payoffs_antithetic = np.maximum(final_prices_antithetic - K, 0)

    payoffs[barrier_hit] = 0
    payoffs_antithetic[barrier_hit_antithetic] = 0

    expected_S = S0 * np.exp(r * T)
    var_S = np.var(final_prices, ddof=1)
    var_V = np.var(payoffs, ddof=1)
    cov_VS = np.cov(payoffs, final_prices, ddof=1)[0, 1]

    theta_opt = cov_VS / var_S if var_S > 0 else 0

    var_S_anti = np.var(final_prices, ddof=1)
    var_V_anti = np.var(payoffs_antithetic, ddof=1)
    cov_VS_anti = np.cov(payoffs_antithetic, final_prices_antithetic, ddof=1)[0, 1]

    theta_opt_anti = cov_VS_anti / var_S_anti if var_S_anti > 0 else 0

    # Apply control variates to both sets
    adjusted_payoffs = payoffs + theta_opt * (expected_S - final_prices)
    adjusted_payoffs_antithetic = payoffs_antithetic + theta_opt_anti * (expected_S - final_prices_antithetic)

    # Use antithetic variates by averaging both adjusted paths
    final_payoffs = (adjusted_payoffs + adjusted_payoffs_antithetic) / 2

    # Compute Monte Carlo estimates
    aM = discount_factor * np.mean(final_payoffs)
    bM = discount_factor * np.std(final_payoffs, ddof=1)

    # Compute actual confidence interval bounds
    ci_lower = aM - 1.96 * (bM / np.sqrt(M))
    ci_upper = aM + 1.96 * (bM / np.sqrt(M))
    confidence_interval = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

    # Compute required M for CI width < $0.01
    required_M = int(np.ceil(((2 * 1.96 * bM) / 0.01) ** 2))

    # End timing
    end_time = time.perf_counter()
    comp_time = end_time - start_time  # Computational time for M

    # Estimate time for required_M
    estimated_time_required_M = (required_M / M) * comp_time

    # Compute speedup factor
    speedup_factor = base_time / estimated_time_required_M

    return aM, bM, confidence_interval, required_M, comp_time, estimated_time_required_M, speedup_factor, theta_opt, var_S, cov_VS, var_V

# Run function with given parameters
aM_4a, bM_4a, confidence_interval_4a, required_M_4a, comp_time_4a, estimated_time_required_M_4a, speedup_factor_4a, theta_opt_4a, var_S_4a, cov_VS_4a, var_V_4a  = milstein_barrier_option_control_antithetic(100, 90, 61, 0.01, 0.1, 1.5, 1000, 10000)

labels = [
    r"$a_M$ (\$)",
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)",
    r"$M^{*}$",
    r"$t^{*}$ (seconds)",
    "Speedup Factor"
]

values = [
    f"{aM_4a:.4f}",
    f"{bM_4a:.4f}",
    confidence_interval_4a,
    f"{comp_time_4a:.4f}",
    f"{required_M_4a}",
    f"{estimated_time_required_M_4a:.2f}",
    f"${speedup_factor_4a:.4f} \\times$"
]

# Create DataFrame for results
df_results4a = pd.DataFrame([values], columns=labels)

# Save results to LaTeX format
df_results4a.to_latex("results_part4a.tex", index=False, escape=False, column_format="c" * df_results4a.shape[1])

# Print the DataFrame
print(df_results4a)

def milstein_barrier_option_antithetic_control(S0, K, B, r, sigma, T, N, M):
    dt = T / N  # Time step size
    discount_factor = np.exp(-r * T)  # Discount factor for final payoff

    # Start timing
    start_time = time.perf_counter()

    # Step 1: Apply Antithetic Variates
    Z = np.random.randn(M, N)  # Generate normal random variables
    Z_antithetic = -Z  # Antithetic variates

    S = np.full(M, S0, dtype=np.float64)
    S_antithetic = np.full(M, S0, dtype=np.float64)

    barrier_hit = np.zeros(M, dtype=bool)
    barrier_hit_antithetic = np.zeros(M, dtype=bool)

    for i in range(N):
        dW = np.sqrt(dt) * Z[:, i]
        dW_antithetic = np.sqrt(dt) * Z_antithetic[:, i]

        S = S + r * S * dt + sigma * S * dW + 0.5 * sigma**2 * S * (dW**2 - dt)
        S_antithetic = S_antithetic + r * S_antithetic * dt + sigma * S_antithetic * dW_antithetic + 0.5 * sigma**2 * S_antithetic * (dW_antithetic**2 - dt)

        barrier_hit |= (S < B)
        barrier_hit_antithetic |= (S_antithetic < B)

    # Compute payoffs
    final_prices = S
    final_prices_antithetic = S_antithetic

    payoffs = np.maximum(final_prices - K, 0)
    payoffs_antithetic = np.maximum(final_prices_antithetic - K, 0)

    payoffs[barrier_hit] = 0
    payoffs_antithetic[barrier_hit_antithetic] = 0

    # Use antithetic variates by averaging both paths
    final_payoffs = (payoffs + payoffs_antithetic) / 2

    # Step 2: Apply Control Variates
    expected_S = S0 * np.exp(r * T)
    var_S = np.var(final_prices, ddof=1)
    var_V = np.var(final_payoffs, ddof=1)
    cov_VS = np.cov(final_payoffs, final_prices, ddof=1)[0, 1]

    theta_opt = cov_VS / var_S if var_S > 0 else 0

    # Apply control variates to final payoffs
    adjusted_payoffs = final_payoffs + theta_opt * (expected_S - final_prices)

    # Compute Monte Carlo estimates
    aM = discount_factor * np.mean(adjusted_payoffs)  # Option price estimate
    bM = discount_factor * np.std(adjusted_payoffs, ddof=1)

    # Compute actual confidence interval bounds
    ci_lower = aM - 1.96 * (bM / np.sqrt(M))
    ci_upper = aM + 1.96 * (bM / np.sqrt(M))
    confidence_interval = f"[{ci_lower:.4f}, {ci_upper:.4f}]"

    # Compute required M for CI width < $0.01
    required_M = int(np.ceil(((2 * 1.96 * bM) / 0.01) ** 2))

    # End timing
    end_time = time.perf_counter()
    comp_time = end_time - start_time  # Computational time for M

    # Estimate time for required_M
    estimated_time_required_M = (required_M / M) * comp_time

    # Compute speedup factor
    speedup_factor = base_time / estimated_time_required_M

    return aM, bM, confidence_interval, required_M, comp_time, estimated_time_required_M, speedup_factor, theta_opt, var_S, cov_VS, var_V

aM_4b, bM_4b, confidence_interval_4b, required_M_4b, comp_time_4b, estimated_time_required_M_4b, speedup_factor_4b, theta_opt_4b, var_S_4b, cov_VS_4b, var_V_4b  = milstein_barrier_option_antithetic_control(100, 90, 61, 0.01, 0.1, 1.5, 1000, 10000)

labels = [
    r"$a_M$ (\$)",
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)",
    r"$M^{*}$",
    r"$t^{*}$ (seconds)",
    "Speedup Factor"
]

values = [
    f"{aM_4b:.4f}",
    f"{bM_4b:.4f}",
    confidence_interval_4b,
    f"{comp_time_4b:.4f}",
    f"{required_M_4b}",
    f"{estimated_time_required_M_4b:.2f}",
    f"${speedup_factor_4b:.4f} \\times$"
]

# Create DataFrame for results
df_results4b = pd.DataFrame([values], columns=labels)

# Save results to LaTeX format
df_results4b.to_latex("results_part4b.tex", index=False, escape=False, column_format="c" * df_results4b.shape[1])

# Print the DataFrame
print(df_results4b)

# Part 5
# Results and Discussions
# Define function names for the x-axis labels
function_names = ["Base Case", "Antithetic", "Control", "Control → Antithetic", "Antithetic → Control"]

# Required M values for plotting
required_M_values = [
    required_M_1,
    required_M_2,
    required_M_3,
    required_M_4a,
    required_M_4b
]

# Speedup factors for plotting
speedup_factors = [
    1.0,  # Base MC (reference point, speedup factor = 1)
    speedup_factor_2,
    speedup_factor_3,
    speedup_factor_4a,
    speedup_factor_4b
]

bM_values = [bM_1, bM_2, bM_3, bM_4a, bM_4b]

# Define professional color scheme
colours = ["#0072B2", "#E69F00", "#009E73", "#F0E442", "#CC79A7"]

# Create a 3x2 grid layout to ensure equal subplot sizes
fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(2, 2, figure=fig)

# Top-left plot: Required M
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(function_names, required_M_values, color=colours)
ax1.set_title("Required Number of MC Samples")
ax1.set_ylabel("Required M")
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_xticklabels(function_names, rotation=15, ha="right")
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(bar.get_height())}", 
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Top-right plot: Speedup Factor
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(function_names, speedup_factors, color=colours)
ax2.set_title("Speedup Factor Across Methods")
ax2.set_ylabel("Speedup Factor")
ax2.grid(axis='y', linestyle='--', alpha=0.7)
ax2.set_xticklabels(function_names, rotation=15, ha="right")
for i, v in enumerate(speedup_factors):
    ax2.text(i, v, f"{v:.2f}×", ha='center', va='bottom', fontsize=9, fontweight='bold')

# Bottom-center plot: bM Comparison
# Adding bottom plot manually in a centered position below top two
bottom_ax_width = 0.45
bottom_ax_height = 0.35
bottom_ax_x = 0.5 - bottom_ax_width / 2  # Center horizontally
bottom_ax_y = 0.1  # Bottom vertical position

ax3 = fig.add_axes([bottom_ax_x, bottom_ax_y, bottom_ax_width, bottom_ax_height])

bars3 = ax3.bar(function_names, bM_values, color=colours)
ax3.set_title(r"$b_M$ Values Across Methods")
ax3.set_ylabel(r"$b_M$ (Standard Deviation)")
ax3.grid(axis='y', linestyle='--', alpha=0.7)
ax3.set_xticklabels(function_names, rotation=15, ha="right")
for bar in bars3:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.4f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# Adjust the top two subplots slightly upwards to make spacing more balanced
plt.subplots_adjust(hspace=0.3, top=0.93)

# Save the figure
plt.savefig("threeway_comparison_plots.png", dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

# Estimate for option price with 95% CI < $0.01 for control then antithetic variates with required M
aM_f, bM_f, confidence_interval_f, required_M_f, comp_time_f, estimated_time_required_M_f, speedup_factor_f, theta_opt_f, var_S_f, cov_VS_f, var_V_f  = milstein_barrier_option_control_antithetic(100, 90, 61, 0.01, 0.1, 1.5, 1000, 655463)

labels = [
    r"$a_M$ (\$)",
    r"$b_M$ (\$)",
    r"$CI(\$)$",
    r"$t$ (seconds)"
]

values = [f"{aM_f:.4f}",f"{bM_f:.4f}",confidence_interval_f,f"{comp_time_f}"]

# Create DataFrame for results
df_resultsf = pd.DataFrame([values], columns=labels)

# Save results to LaTeX format
df_resultsf.to_latex("results_partf.tex", index=False, escape=False, column_format="c" * df_resultsf.shape[1])

# Print the DataFrame
print(df_resultsf)
