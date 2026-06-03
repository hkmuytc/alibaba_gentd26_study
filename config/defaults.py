"""
Configuration defaults for the prototype.
"""

# Power model parameters (Fan et al., 2007)
GPU_P_IDLE = 50.0   # watts per GPU at idle
GPU_P_MAX = 300.0   # watts per GPU at full load
CPU_P_IDLE = 100.0  # watts per server at idle
CPU_P_MAX = 250.0   # watts per server at full load
MEM_P_IDLE = 10.0   # watts per memory unit at idle
MEM_P_MAX = 40.0    # watts per memory unit at full load

# Default cluster sizes
DEFAULT_N_GPUS = 100
DEFAULT_N_MACHINES = 100

# Training defaults
DEFAULT_WINDOW_SIZE = 24       # 24 steps * 5min = 2 hours lookback
DEFAULT_EPOCHS = 100
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_TRAIN_RATIO = 0.8

# Model architecture defaults
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 2
DEFAULT_DROPOUT = 0.2
DEFAULT_D_MODEL = 64
DEFAULT_NHEAD = 4
DEFAULT_DIM_FF = 128

# Aggregation
DEFAULT_FREQ_SECONDS = 300  # 5 minutes
