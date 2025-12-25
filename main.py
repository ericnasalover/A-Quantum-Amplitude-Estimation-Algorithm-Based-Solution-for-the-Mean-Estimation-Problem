"""
Quantum Amplitude Estimation for Mean Estimation (Weighted Version Only)
Estimate the mean of any function over any interval

Usage:
    1. Modify the function f(x) below
    2. Set the interval [a, b]
    3. Run the code
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit_aer import Aer

# ============================================================================
# CONFIGURATION - MODIFY THESE
# ============================================================================

def f(x):
    """x²"""
    return x**2  # Change this to any function!
    # Examples:
    # return np.sin(x)
    # return np.exp(x)
    # return x**3 - 2*x

# Domain configuration
INTERVAL_START = 0    # Start of interval
INTERVAL_END = 1      # End of interval
N_DOMAIN_QUBITS = 5   # Number of qubits for discretization (2^n points)

# QPE configuration
N_COUNTING_QUBITS = 5  # Precision: more qubits = better precision (but deeper circuit)
N_SHOTS = 8192         # Number of measurements

# Weighted average configuration
TOP_K_OUTCOMES = 10    # Use top K outcomes for weighted average

# ============================================================================
# Step 1: Discretize the function
# ============================================================================

N_POINTS = 2**N_DOMAIN_QUBITS
x_values = np.linspace(INTERVAL_START, INTERVAL_END, N_POINTS)
f_values = np.array([f(x) for x in x_values])

print("=" * 70)
print("QUANTUM AMPLITUDE ESTIMATION - WEIGHTED VERSION")
print("=" * 70)
print(f"\nFunction: f(x) = {f.__doc__.strip() if f.__doc__ else 'custom function'}")
print(f"Interval: [{INTERVAL_START}, {INTERVAL_END}]")
print(f"Discretization: {N_POINTS} points ({N_DOMAIN_QUBITS} qubits)")
print(f"\nDiscretized values:")
for i, (x, fx) in enumerate(zip(x_values, f_values)):
    print(f"  x[{i}] = {x:.4f}, f(x) = {fx:.4f}")

# ============================================================================
# Step 2: Normalize to [0, 1]
# ============================================================================

f_min = np.min(f_values)
f_max = np.max(f_values)

if abs(f_max - f_min) < 1e-10:
    print("\n⚠ Warning: Function is constant! Mean is trivial.")
    print(f"Mean = {f_min:.6f}")
    exit(0)

g_values = (f_values - f_min) / (f_max - f_min)

print(f"\n{'='*70}")
print(f"NORMALIZATION")
print(f"{'='*70}")
print(f"f_min = {f_min:.6f}")
print(f"f_max = {f_max:.6f}")
print(f"g(x) = (f(x) - {f_min:.4f}) / ({f_max:.4f} - {f_min:.4f})")

# Analytical mean
mean_f_analytical = np.mean(f_values)
mean_g_analytical = np.mean(g_values)

print(f"\nAnalytical results:")
print(f"  Mean of f(x): {mean_f_analytical:.6f}")
print(f"  Mean of g(x) = a: {mean_g_analytical:.6f}")

# ============================================================================
# Step 3: State Preparation |ψ⟩
# ============================================================================

def create_state_preparation(g_values, n_qubits):
    """
    Create |ψ⟩ = (1/√N) Σⱼ [√g(j)|j⟩|1⟩ + √(1-g(j))|j⟩|0⟩]
    """
    N = len(g_values)
    qc = QuantumCircuit(n_qubits + 1, name='A')
    
    # Equal superposition on domain
    for i in range(n_qubits):
        qc.h(i)
    
    # Encode function values as rotations
    for j in range(N):
        if 0 < g_values[j] < 1:
            theta = 2 * np.arcsin(np.sqrt(g_values[j]))
            binary_j = format(j, f'0{n_qubits}b')
            
            # Apply X where bit is 0
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
            
            # Multi-controlled rotation
            qc.mcry(theta, list(range(n_qubits)), n_qubits)
            
            # Undo X gates
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
        
        elif g_values[j] >= 0.999:
            binary_j = format(j, f'0{n_qubits}b')
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
            qc.mcx(list(range(n_qubits)), n_qubits)
            for qubit_idx, bit in enumerate(binary_j):
                if bit == '0':
                    qc.x(qubit_idx)
    
    return qc

print(f"\n{'='*70}")
print(f"BUILDING QUANTUM CIRCUITS")
print(f"{'='*70}")

state_prep = create_state_preparation(g_values, N_DOMAIN_QUBITS)
print(f"\n✓ State preparation circuit created")
print(f"  Depth: {state_prep.depth()}")

# ============================================================================
# Step 4: Verify State Preparation
# ============================================================================

print(f"\n{'='*70}")
print(f"VERIFICATION")
print(f"{'='*70}")

verify_qc = QuantumCircuit(N_DOMAIN_QUBITS + 1, N_DOMAIN_QUBITS + 1)
verify_qc.compose(state_prep, inplace=True)
verify_qc.measure_all()

backend = Aer.get_backend('qasm_simulator')
verify_transpiled = transpile(verify_qc, backend)
job = backend.run(verify_transpiled, shots=10000)
result = job.result()
counts = result.get_counts()

prob_ancilla_1 = sum(count for bitstring, count in counts.items() 
                     if bitstring[0] == '1') / 10000

print(f"\nState preparation verification:")
print(f"  P(ancilla=|1⟩) measured: {prob_ancilla_1:.6f}")
print(f"  Expected (mean of g): {mean_g_analytical:.6f}")
print(f"  Difference: {abs(prob_ancilla_1 - mean_g_analytical):.6f}")

if abs(prob_ancilla_1 - mean_g_analytical) < 0.05:
    print(f"  ✓ Verified!")
else:
    print(f"  ⚠ Large discrepancy detected")

# ============================================================================
# Step 5: Grover Operator
# ============================================================================

def create_grover_operator(state_prep, n_domain):
    """Create Grover operator Q"""
    n_qubits = n_domain + 1
    ancilla = n_domain
    qc = QuantumCircuit(n_qubits, name='Q')
    
    # Mark bad state (ancilla = 0)
    qc.x(ancilla)
    qc.z(ancilla)
    qc.x(ancilla)
    
    # A^(-1)
    qc.compose(state_prep.inverse(), inplace=True)
    
    # Reflect about |0⟩
    for i in range(n_qubits):
        qc.x(i)
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    for i in range(n_qubits):
        qc.x(i)
    
    # A
    qc.compose(state_prep, inplace=True)
    
    return qc

grover_op = create_grover_operator(state_prep, N_DOMAIN_QUBITS)
print(f"\n✓ Grover operator created")
print(f"  Depth: {grover_op.depth()}")

# ============================================================================
# Step 6: QPE Circuit
# ============================================================================

def create_qae_circuit(state_prep, grover_op, n_counting, n_domain):
    """Create Quantum Amplitude Estimation circuit"""
    n_workspace = n_domain + 1
    
    counting = QuantumRegister(n_counting, 'cnt')
    workspace = QuantumRegister(n_workspace, 'wsp')
    c = ClassicalRegister(n_counting, 'c')
    
    qc = QuantumCircuit(counting, workspace, c)
    
    # Initialize workspace
    qc.compose(state_prep, workspace, inplace=True)
    qc.barrier()
    
    # Hadamard on counting
    qc.h(counting)
    qc.barrier()
    
    # Controlled Grover operations
    for j in range(n_counting):
        power = 2**j
        for p in range(power):
            ctrl_q = grover_op.control(1)
            qc.compose(ctrl_q, [counting[j]] + list(workspace), inplace=True)
    
    qc.barrier()
    
    # Inverse QFT
    qft_inv = QFT(n_counting, inverse=True)
    qc.compose(qft_inv, counting, inplace=True)
    qc.barrier()
    
    # Measure
    qc.measure(counting, c)
    
    return qc

M = 2**N_COUNTING_QUBITS
print(f"\n✓ QPE configuration:")
print(f"  Counting qubits: {N_COUNTING_QUBITS}")
print(f"  Precision levels: {M}")

qae_circuit = create_qae_circuit(state_prep, grover_op, N_COUNTING_QUBITS, N_DOMAIN_QUBITS)
print(f"\n✓ QAE circuit created")
print(f"  Total qubits: {qae_circuit.num_qubits}")
print(f"  Circuit depth: {qae_circuit.depth()}")

# ============================================================================
# Step 7: Run Simulation
# ============================================================================

print(f"\n{'='*70}")
print(f"SIMULATION")
print(f"{'='*70}")
print(f"Running with {N_SHOTS} shots...")

qae_transpiled = transpile(qae_circuit, backend, optimization_level=1)
print(f"✓ Circuit transpiled (depth: {qae_transpiled.depth()})")

job = backend.run(qae_transpiled, shots=N_SHOTS)
result = job.result()
counts = result.get_counts()
print(f"✓ Simulation complete!")

# ============================================================================
# Step 8: Post-Processing - WEIGHTED AVERAGE ONLY
# ============================================================================

print(f"\n{'='*70}")
print(f"RESULTS - WEIGHTED AVERAGE METHOD")
print(f"{'='*70}")

# Extract estimates
estimates = []
for bitstring, count in counts.items():
    y = int(bitstring, 2)
    prob = count / N_SHOTS
    theta_a = np.pi * y / M
    a_estimate = np.sin(theta_a)**2
    estimates.append((a_estimate, prob, y, theta_a))

# Sort by probability
estimates.sort(key=lambda x: x[1], reverse=True)

# Display top outcomes
print(f"\nTop {min(TOP_K_OUTCOMES, len(estimates))} measurement outcomes:")
print(f"{'Rank':<6} {'y':<6} {'Prob':<10} {'θ̃ₐ':<12} {'ã':<12}")
print("-" * 56)

for rank, (a, prob, y, theta) in enumerate(estimates[:TOP_K_OUTCOMES]):
    print(f"{rank+1:<6} {y:<6} {prob:<10.4f} {theta:<12.4f} {a:<12.6f}")

# Compute weighted average
top_estimates = estimates[:TOP_K_OUTCOMES]
total_prob = sum(prob for _, prob, _, _ in top_estimates)
weighted_a = sum(a * prob for a, prob, _, _ in top_estimates) / total_prob

print(f"\n{'='*70}")
print(f"WEIGHTED AVERAGE ESTIMATE")
print(f"{'='*70}")
print(f"Using top {TOP_K_OUTCOMES} outcomes (total probability: {total_prob:.4f})")
print(f"\nEstimated amplitude:")
print(f"  ã = {weighted_a:.6f}")
print(f"  True a = {mean_g_analytical:.6f}")
print(f"  Absolute error: {abs(weighted_a - mean_g_analytical):.6f}")
print(f"  Relative error: {abs(weighted_a - mean_g_analytical)/mean_g_analytical*100:.2f}%")

# Convert to mean of f
mean_f_estimate = weighted_a * (f_max - f_min) + f_min

print(f"\n{'='*70}")
print(f"FINAL RESULT - MEAN OF f(x)")
print(f"{'='*70}")
print(f"\n  Analytical mean: {mean_f_analytical:.6f}")
print(f"  Quantum estimate: {mean_f_estimate:.6f}")
print(f"  Absolute error: {abs(mean_f_estimate - mean_f_analytical):.6f}")
print(f"  Relative error: {abs(mean_f_estimate - mean_f_analytical)/mean_f_analytical*100:.2f}%")

# ============================================================================
# Step 9: Visualization
# ============================================================================

print(f"\n{'='*70}")
print(f"CREATING VISUALIZATIONS")
print(f"{'='*70}")

fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# Plot 1: Original function
ax1 = axes[0, 0]
x_continuous = np.linspace(INTERVAL_START, INTERVAL_END, 500)
f_continuous = np.array([f(x) for x in x_continuous])
ax1.plot(x_continuous, f_continuous, 'b-', linewidth=2.5, label='f(x)', zorder=1)
ax1.scatter(x_values, f_values, color='red', s=120, zorder=5, 
           edgecolor='black', linewidth=1.5, label='Discretized points')
ax1.axhline(mean_f_analytical, color='green', linestyle='--', 
           linewidth=2.5, label=f'True mean = {mean_f_analytical:.4f}', alpha=0.7, zorder=2)
ax1.axhline(mean_f_estimate, color='purple', linestyle=':', 
           linewidth=2.5, label=f'QAE estimate = {mean_f_estimate:.4f}', alpha=0.7, zorder=2)
ax1.set_xlabel('x', fontsize=13, fontweight='bold')
ax1.set_ylabel('f(x)', fontsize=13, fontweight='bold')
ax1.set_title(f'Function over [{INTERVAL_START}, {INTERVAL_END}]', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Normalized function
ax2 = axes[0, 1]
bars = ax2.bar(range(N_POINTS), g_values, color='skyblue', edgecolor='black', 
              linewidth=1.5, alpha=0.8)
ax2.axhline(mean_g_analytical, color='red', linestyle='--', 
           linewidth=2.5, label=f'True a = {mean_g_analytical:.4f}')
ax2.axhline(weighted_a, color='purple', linestyle=':', 
           linewidth=2.5, label=f'QAE ã = {weighted_a:.4f}')
ax2.set_xlabel('Index j', fontsize=13, fontweight='bold')
ax2.set_ylabel('g(j)', fontsize=13, fontweight='bold')
ax2.set_title('Normalized Values g(j) ∈ [0,1]', fontsize=15, fontweight='bold')
ax2.set_xticks(range(N_POINTS))
ax2.legend(fontsize=11, loc='best')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0, 1.1])

# Add values on bars
for i, (bar, val) in enumerate(zip(bars, g_values)):
    height = bar.get_height()
    if height > 0.05:  # Only show label if bar is visible
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 3: QPE measurement distribution
ax3 = axes[1, 0]
top_n = min(min(M, 12), len(estimates))  # Show at most 12 outcomes
y_plot = [estimates[i][2] for i in range(top_n)]
prob_plot = [estimates[i][1] for i in range(top_n)]
colors = ['green'] + ['lightcoral']*(min(5, top_n)-1) + ['lightblue']*(top_n-min(5, top_n))

bars3 = ax3.bar(range(top_n), prob_plot, color=colors, edgecolor='black', 
               linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Measurement outcome (rank)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Probability', fontsize=13, fontweight='bold')
ax3.set_title(f'QPE Distribution (top {top_n} outcomes)', fontsize=15, fontweight='bold')
ax3.set_xticks(range(top_n))
ax3.set_xticklabels([f'y={y}' for y in y_plot], fontsize=9, rotation=45)
ax3.grid(True, alpha=0.3, axis='y')

# Add probability labels
for i, (bar, prob) in enumerate(zip(bars3, prob_plot)):
    if i < 8:  # Only label first 8 to avoid clutter
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.3f}', ha='center', va='bottom', fontsize=8)

# Plot 4: Error comparison
ax4 = axes[1, 1]
error_abs = abs(mean_f_estimate - mean_f_analytical)
error_rel = error_abs / abs(mean_f_analytical) * 100

categories = ['Analytical\nTruth', 'Quantum\nEstimate']
values = [mean_f_analytical, mean_f_estimate]
colors_bar = ['green', 'purple']
bars4 = ax4.bar(categories, values, color=colors_bar, edgecolor='black', 
               linewidth=2, alpha=0.7, width=0.6)

ax4.set_ylabel('Mean value of f(x)', fontsize=13, fontweight='bold')
ax4.set_title('Mean Estimation Results', fontsize=15, fontweight='bold')
ax4.axhline(mean_f_analytical, color='green', linestyle='--', 
           linewidth=2, alpha=0.5)
ax4.grid(True, alpha=0.3, axis='y')

# Value labels
for bar, val in zip(bars4, values):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.6f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Add error text
ax4.text(0.5, 0.95, f'Absolute Error: {error_abs:.6f}\nRelative Error: {error_rel:.2f}%',
        transform=ax4.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        ha='center')

plt.suptitle(f'Quantum Amplitude Estimation: Mean of f(x) over [{INTERVAL_START}, {INTERVAL_END}]', 
            fontsize=17, fontweight='bold', y=0.995)
plt.tight_layout()
plt.show()


print(f"\n{'='*70}")
print(f"COMPLETE")
print(f"{'='*70}")
print(f"\nSummary:")
print(f"  Function: f(x) over [{INTERVAL_START}, {INTERVAL_END}]")
print(f"  Discretization: {N_POINTS} points")
print(f"  QPE precision: {M} levels ({N_COUNTING_QUBITS} qubits)")
print(f"  True mean: {mean_f_analytical:.6f}")
print(f"  Estimated mean: {mean_f_estimate:.6f}")
print(f"  Relative error: {error_rel:.2f}%")
print(f"\n✓ Quantum amplitude estimation complete!")
