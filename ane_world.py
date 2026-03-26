import os
import sys
import numpy as np
import pygame

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
except ImportError:
    print("Warning: coremltools not found. The model cannot be built.")

# --- Configuration ---
W_GRID, H_GRID = 64, 64
RENDER_SCALE = 8
W_PX, H_PX = W_GRID * RENDER_SCALE, H_GRID * RENDER_SCALE

CH_STATE = 4    # 0:Food, 1:Energy, 2:Memory, 3:Pheromones
CH_WEIGHTS = 16 # 4x4 weights for the pixel-wise brain
CH_TOTAL = CH_STATE + CH_WEIGHTS

MODEL_PATH = "build/ane_nca_world.mlpackage"

# --- 1. The ANE Native Brain & Physics ---
def build_ane_model():
    """
    Builds the core simulation logic natively for the Apple Neural Engine.
    Implements a self-referential Neural Cellular Automata where every pixel
    applies its own customized weight-channels to its state-channels.
    """
    # 3x3 depthwise kernel: applies a slight blur/diffusion to all channels.
    # This acts as our "Movement" & "Sensing" step organically.
    diffuse_kernel = np.array([
        [[0.05, 0.10, 0.05],
         [0.10, 0.40, 0.10],
         [0.05, 0.10, 0.05]]
    ], dtype=np.float32).reshape(1, 1, 3, 3)
    
    # Broadcast to all channels for depthwise execution
    w_diffuse = np.repeat(diffuse_kernel, CH_TOTAL, axis=0)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CH_TOTAL, H_GRID, W_GRID))])
    def nca_step(world_grid):
        # Step 1: Spatial Diffusion (Sensing and Movement)
        padded = mb.pad(x=world_grid, pad=[0, 0, 0, 0, 1, 1, 1, 1], mode="constant", constant_val=np.float32(0.0))
        sensed = mb.conv(x=padded, weight=w_diffuse, pad_type="valid", groups=CH_TOTAL)

        # Step 2: Extract State (X) and Pixel-wise Weights (W)
        # We manually slice channels to execute X * W per-pixel dynamically.
        X = []
        for i in range(CH_STATE):
            ch = mb.slice_by_index(x=sensed, begin=[0, i, 0, 0], end=[1, i+1, H_GRID, W_GRID])
            X.append(ch)

        # Step 3: Evaluate Brain dynamically mapping State -> Deltas
        deltas = []
        for i in range(CH_STATE):
            neuron_sum = None
            for j in range(CH_STATE):
                w_idx = CH_STATE + (i * CH_STATE) + j
                W_ij = mb.slice_by_index(x=sensed, begin=[0, w_idx, 0, 0], end=[1, w_idx+1, H_GRID, W_GRID])
                
                # Point-wise multiply state channel * weight channel
                term = mb.mul(x=X[j], y=W_ij)
                
                if neuron_sum is None:
                    neuron_sum = term
                else:
                    neuron_sum = mb.add(x=neuron_sum, y=term)
                    
            # Non-linear activation for the biological response
            deltas.append(mb.tanh(x=neuron_sum))

        # Concatenate delta channels back together
        state_delta = mb.concat(values=deltas, axis=1)

        # Extract old state and weights
        old_state = mb.slice_by_index(x=sensed, begin=[0, 0, 0, 0], end=[1, CH_STATE, H_GRID, W_GRID])
        weights   = mb.slice_by_index(x=sensed, begin=[0, CH_STATE, 0, 0], end=[1, CH_TOTAL, H_GRID, W_GRID])

        # Apply physics/biology rules
        # New State = (Old State + Brain Delta)
        # However, we decay it slightly to simulate metabolism / entropy!
        decayed = mb.mul(x=old_state, y=np.float32(0.95))
        new_state = mb.add(x=decayed, y=state_delta)

        # Recombine world
        new_world = mb.concat(values=[new_state, weights], axis=1)
        
        # Clamp to prevent runaway numerical explosions
        new_world = mb.clip(x=new_world, alpha=np.float32(-1.0), beta=np.float32(1.0))
        return new_world

    print(f"Compiling self-referential grid ({CH_TOTAL} channels) for ANE...")
    return ct.convert(
        nca_step, 
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13
    )

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model = build_ane_model()
        model.save(MODEL_PATH)
    return ct.models.MLModel(MODEL_PATH)

# --- 2. Simulation Environment Pipeline ---
def init_world():
    """Generates initial grid with random noise and structured organisms."""
    world = np.zeros((1, CH_TOTAL, H_GRID, W_GRID), dtype=np.float32)
    
    # 1. Random bursts of food
    world[0, 0] = np.random.rand(H_GRID, W_GRID) * 0.5
    
    # 2. Spawn simple "organisms" in the center
    # They have max energy and randomized neural weights to see how they evolve/diffuse!
    cx, cy = W_GRID // 2, H_GRID // 2
    s = 5
    world[0, 1, cy-s:cy+s, cx-s:cx+s] = 1.0 # Max Energy
    
    # Give the organisms completely random genetic weights!
    world[0, CH_STATE:, cy-s:cy+s, cx-s:cx+s] = np.random.randn(CH_WEIGHTS, s*2, s*2) * 2.0
    
    return world

def render_to_surface(world_tensor):
    """Maps tensor channels to RGB and blits to pygame."""
    # Squeeze out batch layer: (CH, H, W)
    t = world_tensor[0]
    
    rgb = np.zeros((H_GRID, W_GRID, 3), dtype=np.uint8)
    
    # Mapping our CA channels to visual colours!
    # Red   = Target 0 (Food)
    # Green = Target 1 (Organism Energy)
    # Blue  = Target 3 (Pheromones/Memory)
    r = np.clip(t[0] * 255.0, 0, 255)
    g = np.clip(t[1] * 255.0, 0, 255)
    b = np.clip(t[3] * 255.0, 0, 255)
    
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b
    
    # Pygame expects (W, H, C)
    rgb = np.transpose(rgb, (1, 0, 2))
    
    surface = pygame.surfarray.make_surface(rgb)
    return surface

# --- 3. Pygame Loop ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((W_PX, H_PX))
    pygame.display.set_caption("Beetle Brain - ANE NCA")
    clock = pygame.time.Clock()

    print("\nLoading CoreML ANE Model...")
    model = get_model()
    
    world = init_world()
    
    running = True
    print("\nSimulation Started! Control+C or close window to end.")
    print(" - Red   = Food")
    print(" - Green = Organism Energy")
    print(" - Blue  = Pheromones")
    print(" -> CLICK to drop new organisms and randomized genetic instructions!")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Click to drop an "energy bomb" + random weights into the sim!
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                gx, gy = x // RENDER_SCALE, y // RENDER_SCALE
                
                # Drop bounded square of max energy and new random DNA
                y0, y1 = max(0, gy-2), min(H_GRID, gy+3)
                x0, x1 = max(0, gx-2), min(W_GRID, gx+3)
                
                world[0, 1, y0:y1, x0:x1] = 1.0 # Max Energy
                dyn_h, dyn_w = y1 - y0, x1 - x0
                world[0, CH_STATE:, y0:y1, x0:x1] = np.random.randn(CH_WEIGHTS, dyn_h, dyn_w) * 5.0

        # Run ANE simulation tick!
        out = model.predict({"world_grid": world})
        world = list(out.values())[0]
        
        # Extra physics check: Food regenerates globally (done in numpy for fast RNG)
        world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.05
        world = np.clip(world, -1.0, 1.0)
        
        # Render Process
        surf = render_to_surface(world)
        surf_scaled = pygame.transform.scale(surf, (W_PX, H_PX))
        
        screen.blit(surf_scaled, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        pygame.display.set_caption(f"Beetle Brain - ANE NCA | {clock.get_fps():.0f} FPS")

    pygame.quit()

if __name__ == "__main__":
    main()