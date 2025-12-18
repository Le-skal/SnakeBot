"""Algorithms module - Contains AI algorithms for Snake"""
from .hamilton_cycle import HamiltonianSnakePlanner, visualize_cycle, generate_hamiltonian_cycle

# Backward compatibility alias
HamiltonPathPlanner = HamiltonianSnakePlanner

__all__ = ['HamiltonianSnakePlanner', 'HamiltonPathPlanner', 'visualize_cycle', 'generate_hamiltonian_cycle']
