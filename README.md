# Monte Carlo Barrier Option Pricing

## MSc Financial Mathematics – Computational Finance Project (University of Leeds)

This repository contains a Python implementation and accompanying report for pricing a down-and-out European barrier call option using Monte Carlo simulation.

The project investigates variance reduction techniques to improve computational efficiency while maintaining pricing accuracy.

---

## Project Overview

The implementation uses the Milstein discretisation of a stochastic differential equation to simulate stock price paths before pricing a down-and-out barrier option using Monte Carlo methods.

Several variance reduction techniques are implemented and compared, including:

- Standard Monte Carlo simulation
- Antithetic variates
- Fine-tuned control variates
- Combined antithetic and control variate methods

Each approach is evaluated in terms of pricing accuracy, confidence interval width, computational time and variance reduction.

---

## Key Features

- Milstein discretisation for stochastic differential equations
- Monte Carlo simulation for barrier option pricing
- Antithetic variates
- Fine-tuned control variates
- Confidence interval estimation
- Computational efficiency analysis
- Comparison of variance reduction techniques

---

## Repository Contents

- `monte_carlo_barrier_option_pricing.py` – Python implementation
- `Monte Carlo Barrier Option Pricing Report.pdf` – Technical report describing the methodology and results

---

## Technologies

- Python
- NumPy
- Matplotlib

---

## Key Outcome

Fine-tuned control variates produced the largest computational speed-up, while combining control variates with antithetic variates achieved the lowest estimator variance and required the fewest Monte Carlo paths for the desired confidence interval.

---

*Academic coursework completed as part of the MSc Financial Mathematics programme at the University of Leeds.*
