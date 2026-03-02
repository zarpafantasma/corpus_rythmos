#!/usr/bin/env python3
"""
S4: Sierpiński Fractal Grid - Empirical α Anchoring
====================================================
Phase 2: Red Team Corrected Pipeline (3D Topology)

From "RTM Unified Field Framework" - Section 4.4.1
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 3D SIERPIŃSKI TETRAHEDRON CONSTRUCTION (RED TEAM FIXED)
# =============================================================================

def build_3d_sierpinski_graph(generation: int) -> tuple:
    """
    Build 3D Sierpiński tetrahedron (sponge) using exact geometry.
    This corrects the 2D topology mismatch.
    """
    if generation == 0:
        graph = {0: [1, 2, 3], 1: [0, 2, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
        pos = {
            0: (0.0, 0.0, 0.61237),
            1: (-0.5, -0.28867, -0.20412),
            2: (0.5, -0.28867, -0.20412),
            3: (0.0, 0.57735, -0.20412)
        }
        return graph, pos
        
    vertices = set()
    edges = set()
    
    def add_tetra(p1, p2, p3, p4, level):
        if level == 0:
            v1, v2, v3, v4 = p1, p2, p3, p4
            vertices.update([v1, v2, v3, v4])
            edges.update([
                (min(v1, v2), max(v1, v2)), (min(v1, v3), max(v1, v3)), (min(v1, v4), max(v1, v4)),
                (min(v2, v3), max(v2, v3)), (min(v2, v4), max(v2, v4)), (min(v3, v4), max(v3, v4))
            ])
        else:
            m12 = tuple(np.round((np.array(p1)+np.array(p2))/2, 5))
            m13 = tuple(np.round((np.array(p1)+np.array(p3))/2, 5))
            m14 = tuple(np.round((np.array(p1)+np.array(p4))/2, 5))
            m23 = tuple(np.round((np.array(p2)+np.array(p3))/2, 5))
            m24 = tuple(np.round((np.array(p2)+np.array(p4))/2, 5))
            m34 = tuple(np.round((np.array(p3)+np.array(p4))/2, 5))
            
            add_tetra(p1, m12, m13, m14, level-1)
            add_tetra(m12, p2, m23, m24, level-1)
            add_tetra(m13, m23, p3, m34, level-1)
            add_tetra(m14, m24, m34, p4, level-1)

    p1 = (0.0, 0.0, 0.61237)
    p2 = (-0.5, -0.28867, -0.20412)
    p3 = (0.5, -0.28867, -0.20412)
    p4 = (0.0, 0.57735, -0.20412)
    
    add_tetra(p1, p2, p3, p4, generation)
    
    vertex_list = sorted(vertices)
    pos_to_id = {v: i for i, v in enumerate(vertex_list)}
    
    graph = defaultdict(list)
    for v1, v2 in edges:
        id1, id2 = pos_to_id[v1], pos_to_id[v2]
        graph[id1].append(id2)
        graph[id2].append(id1)
        
    return dict(graph), {pos_to_id[v]: v for v in vertex_list}

# =============================================================================
# RANDOM WALK SIMULATION
# =============================================================================

def random_walk_mfpt(graph: dict, start: int, targets: set, max_steps: int = 200000) -> int:
    current = start
    steps = 0
    while current not in targets and steps < max_steps:
        neighbors = graph[current]
        current = np.random.choice(neighbors)
        steps += 1
    return steps

def measure_mfpt(graph: dict, center: int, boundary: set, n_walks: int = 500) -> float:
    times = []
    for _ in range(n_walks):
        times.append(random_walk_mfpt(graph, center, boundary))
    return np.mean(times), np.std(times)

def estimate_alpha_sierpinski(generations: list, n_walks: int = 500) -> dict:
    results = {'generation': [], 'L': [], 'n_vertices': [], 'mfpt_mean': [], 'mfpt_std': []}
    
    for g in generations:
        print(f"  Generation {g}...")
        graph, positions = build_3d_sierpinski_graph(g)
        n_vertices = len(graph)
        L = 2**g
        
        # Center vertex
        center = min(positions.keys(), key=lambda v: positions[v][0]**2 + positions[v][1]**2 + positions[v][2]**2)
        
        # Boundary vertices
        corners = [
            (0.0, 0.0, 0.61237),
            (-0.5, -0.28867, -0.20412),
            (0.5, -0.28867, -0.20412),
            (0.0, 0.57735, -0.20412)
        ]
        boundary = set()
        for c in corners:
            boundary.add(min(positions.keys(), key=lambda v: (positions[v][0]-c[0])**2 + (positions[v][1]-c[1])**2 + (positions[v][2]-c[2])**2))
            
        mfpt_mean, mfpt_std = measure_mfpt(graph, center, boundary, n_walks)
        
        results['generation'].append(g)
        results['L'].append(L)
        results['n_vertices'].append(n_vertices)
        results['mfpt_mean'].append(mfpt_mean)
        results['mfpt_std'].append(mfpt_std)
        
    valid_idx = [i for i, t in enumerate(results['mfpt_mean']) if t > 0]
    log_L = np.log([results['L'][i] for i in valid_idx])
    log_T = np.log([results['mfpt_mean'][i] for i in valid_idx])
    
    coeffs = np.polyfit(log_L, log_T, 1)
    results['alpha'] = coeffs[0]
    results['log_L'] = np.log(results['L'])
    results['log_T'] = np.log([max(t, 0.1) for t in results['mfpt_mean']])
    return results

def create_plots(results: dict, output_dir: str):
    fig = plt.figure(figsize=(14, 12))
    
    # 3D Gasket
    ax1 = fig.add_subplot(221, projection='3d')
    g = 2
    graph, pos = build_3d_sierpinski_graph(g)
    for v, neighbors in graph.items():
        if v in pos:
            x1, y1, z1 = pos[v]
            for n in neighbors:
                if n in pos and n > v:
                    x2, y2, z2 = pos[n]
                    ax1.plot([x1, x2], [y1, y2], [z1, z2], 'b-', linewidth=0.5, alpha=0.5)
    xs = [pos[v][0] for v in pos]; ys = [pos[v][1] for v in pos]; zs = [pos[v][2] for v in pos]
    ax1.scatter(xs, ys, zs, c='blue', s=10)
    ax1.set_title(f'3D Sierpiński Tetrahedron (Gen {g})', fontsize=14)
    ax1.axis('off')
    
    # MFPT scaling
    ax2 = fig.add_subplot(222)
    L = np.array(results['L']); T = np.array(results['mfpt_mean']); T_std = np.array(results['mfpt_std'])
    ax2.errorbar(L, T, yerr=T_std, fmt='bo-', markersize=8, linewidth=2, capsize=5, label='Simulation')
    L_fit = np.logspace(np.log10(L[0]), np.log10(L[-1]), 50)
    T_fit = np.exp(results['log_T'][0]) * (L_fit / L[0])**results['alpha']
    ax2.plot(L_fit, T_fit, 'r--', linewidth=2, label=f'Fit: alpha = {results["alpha"]:.2f}')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('System size L', fontsize=12); ax2.set_ylabel('Mean First-Passage Time <T>', fontsize=12)
    ax2.set_title('MFPT Scaling on 3D Sierpiński Gasket', fontsize=14)
    ax2.legend(); ax2.grid(True, alpha=0.3, which='both')
    
    # RTM Bands
    ax3 = fig.add_subplot(223)
    bands = {'Diffusive': (2.0, 2.0), 'Small-world': (2.26, 2.26), 'Hierarchical': (2.47, 2.61), 'Holographic': (2.61, 2.72), 'Fractal': (2.58, 2.63)}
    y_pos = np.arange(len(bands))
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    for i, (name, (low, high)) in enumerate(bands.items()):
        ax3.barh(i, high - low, left=low, height=0.6, color=colors[i], alpha=0.7, label=name)
        ax3.text(high + 0.02, i, f'{low:.2f}-{high:.2f}', va='center', fontsize=10)
    ax3.axvline(x=results['alpha'], color='black', linewidth=2, linestyle='--', label=f'Measured: {results["alpha"]:.2f}')
    ax3.set_yticks(y_pos); ax3.set_yticklabels(list(bands.keys()))
    ax3.set_xlabel('RTM Exponent alpha', fontsize=12); ax3.set_title('Comparison with RTM Bands', fontsize=14)
    ax3.legend(loc='upper right'); ax3.set_xlim(1.9, 2.9); ax3.grid(True, alpha=0.3, axis='x')
    
    # Residuals
    ax4 = fig.add_subplot(224)
    T_predicted = np.exp(results['log_T'][0]) * (L / L[0])**results['alpha']
    residuals = (T - T_predicted) / T_predicted * 100
    ax4.bar(results['generation'], residuals, color='green', alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=1)
    ax4.set_xlabel('Generation g', fontsize=12); ax4.set_ylabel('Residual (%)', fontsize=12)
    ax4.set_title('Fit Residuals', fontsize=14); ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'S4_sierpinski_fractal_fixed.png'), dpi=150)
    plt.close()

def main():
    print("=" * 70)
    print("S4: Sierpiński Fractal Grid - RED TEAM FIXED")
    print("=" * 70)
    output_dir = "output_S4_fixed"
    os.makedirs(output_dir, exist_ok=True)
    
    generations = [2, 3, 4, 5, 6]
    n_walks = 300
    results = estimate_alpha_sierpinski(generations, n_walks)
    
    print(f"\nFitted alpha = {results['alpha']:.3f}")
    print("Theoretical 3D: alpha ≈ 2.585")
    
    df = pd.DataFrame({'generation': results['generation'], 'L': results['L'], 'n_vertices': results['n_vertices'], 'mfpt_mean': results['mfpt_mean'], 'mfpt_std': results['mfpt_std']})
    df.to_csv(os.path.join(output_dir, 'S4_sierpinski_data_fixed.csv'), index=False)
    create_plots(results, output_dir)
    print(f"\nOutputs saved to: {output_dir}/")

if __name__ == "__main__":
    main()
