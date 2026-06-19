"""
tns: Tensor Network States — Project A4 implementation.

Loschmidt echo and dynamical phase transitions via infinite Time-Evolving
Block Decimation (iTEBD) on the Transverse Field Ising Model (TFIM).

Modules (built incrementally):
    svd_utils   : Phase 0 — truncated SVD, entropy, Schmidt decomposition
    vidal_state : Phase 1 — Vidal canonical form (Gamma-Lambda)
    gates       : Phase 2 — Trotter gate exp(-i h dt)
    tebd        : Phase 3 — gate application in Vidal form
    transfer    : Phase 5 — transfer matrix and Loschmidt rate lambda(t)
    simulate    : Phase 4/5 — main iTEBD loop
"""
