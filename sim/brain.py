"""Thin wrapper around the CoreML/numpy brain for sim-level use."""
from sim.config import MAX_POP, N_INPUTS, N_HIDDEN, N_OUTPUTS
from brain.coreml_brain import init_brain
from brain.coreml_sense_brain import init_sense_brain


def init_ane():
    brain_ok = init_brain(MAX_POP, N_INPUTS, N_HIDDEN, N_OUTPUTS)
    init_sense_brain()
    return brain_ok
