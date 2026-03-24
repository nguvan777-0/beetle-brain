"""Thin wrapper around the CoreML/numpy brain for sim-level use."""
from sim.config import MAX_POP, N_INPUTS, N_HIDDEN, N_OUTPUTS
from brain.coreml_brain import init_brain, run_brain


def init_ane():
    return init_brain(MAX_POP, N_INPUTS, N_HIDDEN, N_OUTPUTS)
