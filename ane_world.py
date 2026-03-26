import os
import sys
import time
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

CH_FOOD = 1
CH_ENERGY = 1
CH_WEIGHTS = 15 # 3 senses * 5 intentions = 15 parameters
CH_ORG = CH_ENERGY + CH_WEIGHTS
CH_TOTAL = CH_FOOD + CH_ORG

MODEL_PATH = "build/ane_nca_world.mlpackage"

# --- Directional Kernels for Discrete Shifting ---
# We represent 5 discrete choices: 0:Stay, 1:North, 2:South, 3:East, 4:West
# To process movement, every pixel "pulls" the state of agents wanting to enter it.
def create_pull_kernels():
    k_stay = np.zeros((1, 1, 3, 3), dtype=np.float32); k_stay[0,0,1,1] = 1.0     
    k_pull_S = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_S[0,0,0,1] = 1.0 
    k_pull_N = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_N[0,0,2,1] = 1.0 
    k_pull_W = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_W[0,0,1,2] = 1.0 
    k_pull_E = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_E[0,0,1,0] = 1.0 
    return [k_stay, k_pull_S, k_pull_N, k_pull_W, k_pull_E]

def mb_circular_pad(x):
    """
    Manually pads a tensor (1, C, H, W) circularly by 1 pixel using ANE-friendly slice and concat.
    """
    # Pad Width (axis 3)
    left_pad = mb.slice_by_index(x=x, begin=[0,0,0,W_GRID-1], end=[1,0,H_GRID,W_GRID], 
                                 begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
    right_pad = mb.slice_by_index(x=x, begin=[0,0,0,0], end=[1,0,H_GRID,1], 
                                  begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
    padded_w = mb.concat(values=[left_pad, x, right_pad], axis=3)

    # Pad Height (axis 2) Note: padded_w is now W_GRID+2
    top_pad = mb.slice_by_index(x=padded_w, begin=[0,0,H_GRID-1,0], end=[1,0,H_GRID,0], 
                                begin_mask=[True,True,False,True], end_mask=[True,True,False,True])
    bottom_pad = mb.slice_by_index(x=padded_w, begin=[0,0,0,0], end=[1,0,1,0], 
                                   begin_mask=[True,True,False,True], end_mask=[True,True,False,True])
    
    return mb.concat(values=[top_pad, padded_w, bottom_pad], axis=2)

def build_discrete_nca_model():
    """Builds a discrete Shift-and-Mask agent simulator on the Apple Neural Engine."""
    kernels = create_pull_kernels()
    pull_k_org = [np.repeat(k, CH_ORG, axis=0) for k in kernels]     # Shift the 16 organism channels
    pull_k_int = [np.repeat(k, 5, axis=0) for k in kernels]          # Shift the 5 intention output channels
    
    # Simple uniform blur for food sensing
    k_blur = np.ones((1, 1, 3, 3), dtype=np.float32) / 9.0

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, CH_TOTAL, H_GRID, W_GRID))])
    def nca_step(world):
        # 1. SLICE WORLD CHANNELS
        food_layer = mb.slice_by_index(x=world, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        org_layer  = mb.slice_by_index(x=world, begin=[0,1,0,0], end=[1,CH_TOTAL,H_GRID,W_GRID])
        energy     = mb.slice_by_index(x=org_layer, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        weights    = mb.slice_by_index(x=org_layer, begin=[0,1,0,0], end=[1,CH_ORG,H_GRID,W_GRID])
        
        # 2. SENSING
        # Use our manual Torus wrap using slice/concat so Core ML compiles happily
        pad_food = mb_circular_pad(food_layer)
        blur_food = mb.conv(x=pad_food, weight=k_blur, pad_type="valid")
        senses = [food_layer, blur_food, energy] # 3 Inputs to the neural net
        
        # 3. ORGANISM BRAIN (Map 3 Senses -> 5 Intentions)
        intent_channels = []
        for out_idx in range(5):
            net = None
            for in_idx in range(3):
                w_idx = (out_idx * 3) + in_idx
                # Dynamic pixel-local weights acting as a dense layer
                w_ch = mb.slice_by_index(x=weights, begin=[0,w_idx,0,0], end=[1,w_idx+1,H_GRID,W_GRID])
                term = mb.mul(x=senses[in_idx], y=w_ch)
                if net is None: net = term
                else: net = mb.add(x=net, y=term)
                
            # Break ties purely deterministically by adding a tiny unique offset to each directional channel
            tie_breaker = np.float32(out_idx * 0.001)
            net = mb.add(x=net, y=tie_breaker)
            intent_channels.append(net)
            
        intent_scores = mb.concat(values=intent_channels, axis=1) # Shape: (1, 5, H, W)
        
        # 4. CHOOSE ONLY 1 DISCRETE ACTION PER PIXEL
        max_score = mb.reduce_max(x=intent_scores, axes=[1], keep_dims=True)
        is_max = mb.equal(x=intent_scores, y=max_score)
        intent_1hot = mb.cast(x=is_max, dtype="fp32")   
        
        # 5. SHIFT-AND-MASK TO RESOLVE MOVEMENT
        # Mode "circular" creates a Torus geometry: organisms walking off the Right edge
        # are pulled into the Left edge perfectly seamlessly.
        pad_org = mb_circular_pad(org_layer)
        pad_int = mb_circular_pad(intent_1hot)
        
        cands_org = []
        cands_valid = []
        
        for d in range(5):
            # Pull the neighbor's state and intention into this pixel
            sh_org = mb.conv(x=pad_org, weight=pull_k_org[d], pad_type="valid", groups=CH_ORG)
            sh_int = mb.conv(x=pad_int, weight=pull_k_int[d], pad_type="valid", groups=5)
            # Did the neighbor we pulled actually *choose* to move in direction 'd'?
            v = mb.slice_by_index(x=sh_int, begin=[0,d,0,0], end=[1,d+1,H_GRID,W_GRID])
            
            cands_org.append(sh_org)
            cands_valid.append(v)
            
        # Collision resolution: Priority ladder (Stay > N > S > E > W)
        final_org = None
        cum_w = None
        for d in range(5):
            # W marks if this candidate is the single winner permitted to enter the cell
            if d == 0:
                w = cands_valid[0]
                cum_w = w
            else:
                w = mb.mul(x=cands_valid[d], y=mb.sub(x=np.float32(1.0), y=cum_w))
                cum_w = mb.add(x=cum_w, y=w)
            # Add winner into the new state (Losers are multiplied by 0.0)
            term = mb.mul(x=cands_org[d], y=w)
            if final_org is None: final_org = term
            else: final_org = mb.add(x=final_org, y=term)
            
        # 6. SURVIVAL & METABOLISM
        shifted_energy = mb.slice_by_index(x=final_org, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        shifted_weights = mb.slice_by_index(x=final_org, begin=[0,1,0,0], end=[1,CH_ORG,H_GRID,W_GRID])
        
        # Burn energy every tick
        post_move_energy = mb.sub(x=shifted_energy, y=np.float32(0.02))
        is_there = mb.cast(x=mb.greater(x=post_move_energy, y=np.float32(0.0)), dtype="fp32")
        
        # If alive, eat the food we are standing on (up to 0.7 units)
        can_eat = mb.mul(x=food_layer, y=is_there)
        food_eaten = mb.clip(x=can_eat, alpha=np.float32(0.0), beta=np.float32(0.7))
        
        # Update biological energy 
        new_energy = mb.add(x=post_move_energy, y=food_eaten)
        new_energy = mb.clip(x=new_energy, alpha=np.float32(0.0), beta=np.float32(1.0))
        new_food = mb.sub(x=food_layer, y=food_eaten)
        
        # Final death mask: IF energy <= 0, weights get zeroed perfectly clean!
        is_alive = mb.cast(x=mb.greater(x=new_energy, y=np.float32(0.0)), dtype="fp32")
        final_energy = mb.mul(x=new_energy, y=is_alive)
        final_weights = mb.mul(x=shifted_weights, y=is_alive)
        
        # Reconstruct exactly matching channels
        next_org = mb.concat(values=[final_energy, final_weights], axis=1)
        next_world = mb.concat(values=[new_food, next_org], axis=1)
        
        return next_world

    print(f"Compiling Discrete Physics NCA Engine ({CH_TOTAL} channels) for ANE...")
    return ct.convert(
        nca_step, 
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.macOS13
    )

def get_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model = build_discrete_nca_model()
        model.save(MODEL_PATH)
    return ct.models.MLModel(MODEL_PATH)

# --- Runtime & Pygame ---
def init_world():
    world = np.zeros((1, CH_TOTAL, H_GRID, W_GRID), dtype=np.float32)
    world[0, 0] = np.random.rand(H_GRID, W_GRID) * 0.3    # Wild food
    return world

def drop_organism(world, gx, gy):
    """Spawns an organism with fully random brain weights."""
    world[0, 1, gy, gx] = 1.0 # Max Energy
    world[0, 2:, gy, gx] = np.random.randn(CH_WEIGHTS) * 4.0 # Random 15 weights

def main():
    headless_ticks = None
    if len(sys.argv) > 1:
        try:
            headless_ticks = int(sys.argv[1])
        except ValueError:
            pass

    is_headless = headless_ticks is not None

    if not is_headless:
        pygame.init()
        screen = pygame.display.set_mode((W_PX, H_PX))
        pygame.display.set_caption("Beetle Brain - Discrete ANE")
        clock = pygame.time.Clock()

    model = get_model()
    world = init_world()
    
    # Spawn 50 starters for a good test
    for _ in range(50):
        drop_organism(world, np.random.randint(W_GRID), np.random.randint(H_GRID))

    if is_headless:
        print(f"\nRunning simulation headless for {headless_ticks} ticks...")
        t0 = time.time()
        for i in range(headless_ticks):
            out = model.predict({"world": world})
            world = list(out.values())[0]
            
            world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.015
            world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
            
            if i % 100 == 0:
                pop = (world[0, 1] > 0).sum()
                print(f"Tick {i:4d} | Population: {pop}")
        
        t1 = time.time()
        fps = headless_ticks / (t1 - t0)
        pop = (world[0, 1] > 0).sum()
        print(f"\nFinished. {headless_ticks} ticks in {t1-t0:.2f}s ({fps:.1f} ticks/sec). Final pop: {pop}")
        return

    running = True
    print("\nSimulation Started!")
    print(" - Clicking spawns an organism. They now step exactly without smearing!")
    print(" - Red   = Food")
    print(" - Green = Organism")
    print(" - Blue  = Neural Weight [0] (Vision)")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                gx = np.clip(event.pos[0] // RENDER_SCALE, 0, W_GRID - 1)
                gy = np.clip(event.pos[1] // RENDER_SCALE, 0, H_GRID - 1)
                # Drop an organism with random genes
                drop_organism(world, gx, gy)
                # Drop some large food nearby
                world[0, 0, max(0,gy-2):min(H_GRID,gy+2), max(0,gx-2):min(W_GRID,gx+2)] += 1.0

        # Run strict 1-tick pass on Neural Engine
        out = model.predict({"world": world})
        world = list(out.values())[0]
        
        # Background physics: global food regeneration bounds
        world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.015
        world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
        
        # Render visual channels
        t = world[0]
        rgb = np.zeros((H_GRID, W_GRID, 3), dtype=np.uint8)
        
        # Dim green for food so it looks like background algae
        rgb[..., 1] = np.clip(t[0] * 100.0, 0, 255) 
        
        org_mask = t[1] > 0
        # Draw organisms
        rgb[org_mask, 0] = np.clip(t[1][org_mask] * 255.0, 80, 255) # Red base from energy
        rgb[org_mask, 1] = np.clip(t[1][org_mask] * 255.0, 80, 255) # Green base from energy
        rgb[org_mask, 2] = np.clip((t[2][org_mask] + 2.0) * 60.0, 0, 255) # Blue varies by genes
        
        rgb = np.transpose(rgb, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rgb)
        surf_scaled = pygame.transform.scale(surf, (W_PX, H_PX))
        
        screen.blit(surf_scaled, (0, 0))
        pygame.display.flip()
        
        clock.tick(60)
        pygame.display.set_caption(f"Beetle Brain - ANE Discrete Evolution | {clock.get_fps():.0f} FPS")

    pygame.quit()

if __name__ == "__main__":
    main()