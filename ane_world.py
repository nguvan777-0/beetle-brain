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
RENDER_SCALE = 12
W_PX, H_PX = W_GRID * RENDER_SCALE, H_GRID * RENDER_SCALE
HUD_WIDTH = 320

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
    # Intent 1: Move North -> We pull from the cell BELOW us (Bottom)
    k_pull_from_B = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_B[0,0,2,1] = 1.0 
    # Intent 2: Move South -> We pull from the cell ABOVE us (Top)
    k_pull_from_T = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_T[0,0,0,1] = 1.0 
    # Intent 3: Move East -> We pull from the cell to our LEFT
    k_pull_from_L = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_L[0,0,1,0] = 1.0 
    # Intent 4: Move West -> We pull from the cell to our RIGHT
    k_pull_from_R = np.zeros((1, 1, 3, 3), dtype=np.float32); k_pull_from_R[0,0,1,2] = 1.0 
    return [k_stay, k_pull_from_B, k_pull_from_T, k_pull_from_L, k_pull_from_R]

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

    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, CH_TOTAL, H_GRID, W_GRID)),
        mb.TensorSpec(shape=(1, CH_WEIGHTS, H_GRID, W_GRID))
    ])
    def nca_step(world, mutation):
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
                
            intent_channels.append(net)
            
        # 4. CHOOSE ONLY 1 DISCRETE ACTION PER PIXEL (STRICTLY 1-HOT TO PREVENT CLONING)
        intent_scores = mb.concat(values=intent_channels, axis=1) # Shape: (1, 5, H, W)
        
        # Use reduce_argmax to strictly break ties and prevent FP16 duplicates!
        best_idx = mb.reduce_argmax(x=intent_scores, axis=1) # Shape: (1, H, W)
        best_idx = mb.cast(x=best_idx, dtype="fp32")
        best_idx_expanded = mb.expand_dims(x=best_idx, axes=[1]) # Shape: (1, 1, H, W)
        
        onehot_channels = []
        for d in range(5):
            d_val = np.array([[[[d]]]], dtype=np.float32)
            c = mb.cast(x=mb.equal(x=best_idx_expanded, y=d_val), dtype="fp32")
            onehot_channels.append(c)
            
        raw_intent_1hot = mb.concat(values=onehot_channels, axis=1) # Shape: (1, 5, H, W)
        
        # Prevent completely empty background cells from generating "ghost" movement intentions
        # that secretly steal priority in the tensor collision ladder and delete living organisms!
        has_energy = mb.cast(x=mb.greater(x=energy, y=np.float32(0.0)), dtype="fp32")
        intent_1hot = mb.mul(x=raw_intent_1hot, y=has_energy)
        
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
        
        # Burn energy every tick (lowered to 0.01 so they have longer life spans to hunt)
        post_move_energy = mb.sub(x=shifted_energy, y=np.float32(0.01))
        is_there = mb.cast(x=mb.greater(x=post_move_energy, y=np.float32(0.0)), dtype="fp32")
        
        # If alive, eat the food we are standing on (eat MORE per tick so they max out fully)
        can_eat = mb.mul(x=food_layer, y=is_there)
        food_eaten = mb.clip(x=can_eat, alpha=np.float32(0.0), beta=np.float32(0.9))
        
        # Update biological energy 
        new_energy = mb.add(x=post_move_energy, y=food_eaten)
        new_energy = mb.clip(x=new_energy, alpha=np.float32(0.0), beta=np.float32(1.0))
        new_food = mb.sub(x=food_layer, y=food_eaten)
        
        # Basic death mask before reproduction
        is_alive_now = mb.cast(x=mb.greater(x=new_energy, y=np.float32(0.0)), dtype="fp32")
        base_energy = mb.mul(x=new_energy, y=is_alive_now)
        base_weights = mb.mul(x=shifted_weights, y=is_alive_now)

        # 7. MITOSIS (Reproduction & Mutation)
        # Parent decides to reproduce if energy > 0.8
        can_reproduce = mb.cast(x=mb.greater(x=base_energy, y=np.float32(0.8)), dtype="fp32")
        
        # Parent loses half energy if reproducing
        cost = mb.mul(x=base_energy, y=mb.mul(x=can_reproduce, y=np.float32(0.5)))
        parent_energy = mb.sub(x=base_energy, y=cost)
        
        # Offspring payload: gets the cost energy, parent's weights + random mutation noise
        mutated_weights = mb.add(x=base_weights, y=mutation)
        offspring_w_send = mb.mul(x=mutated_weights, y=can_reproduce)
        offspring_e_send = cost
        
        # Shift offspring to the East! (Kernel 4 is k_pull_E which pulls from West, meaning moving East)
        offspring_org_send = mb.concat(values=[offspring_e_send, offspring_w_send], axis=1)
        pad_offspring = mb_circular_pad(offspring_org_send)
        inbound_offspring = mb.conv(x=pad_offspring, weight=pull_k_org[4], pad_type="valid", groups=CH_ORG)
        
        # Offspring only survives if it lands on a cell that parent_energy currently evaluates as empty
        is_empty = mb.cast(x=mb.equal(x=parent_energy, y=np.float32(0.0)), dtype="fp32")
        landed_offspring = mb.mul(x=inbound_offspring, y=is_empty)
        
        landed_e = mb.slice_by_index(x=landed_offspring, begin=[0,0,0,0], end=[1,1,H_GRID,W_GRID])
        landed_w = mb.slice_by_index(x=landed_offspring, begin=[0,1,0,0], end=[1,CH_ORG,H_GRID,W_GRID])
        
        # Combine parent and landed offspring
        final_energy = mb.add(x=parent_energy, y=landed_e)
        
        # Ensure parent weights are strictly zeroed if dead
        parent_alive = mb.cast(x=mb.greater(x=parent_energy, y=np.float32(0.0)), dtype="fp32")
        kept_w = mb.mul(x=base_weights, y=parent_alive)
        final_weights = mb.add(x=kept_w, y=landed_w)
        
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
        os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
        pygame.init()
        screen = pygame.display.set_mode((W_PX + HUD_WIDTH, H_PX))
        pygame.display.set_caption("Beetle Brain - Discrete ANE")
        
        # Load fonts
        try:
            font = pygame.font.SysFont("courier", 14, bold=True)
            font_large = pygame.font.SysFont("courier", 24, bold=True)
        except:
            font = pygame.font.SysFont(None, 14, bold=True)
            font_large = pygame.font.SysFont(None, 24, bold=True)
            
        clock = pygame.time.Clock()
        
        # UI State
        import collections
        pop_history = collections.deque(maxlen=280)
        
        # Deterministic color projection for 15 genes to RGB
        np.random.seed(42)
        color_proj = np.random.rand(15, 3)
        color_proj = color_proj / color_proj.sum(axis=0)
        np.random.seed(int(time.time()))

    model = get_model()
    world = init_world()
    
    # Spawn 50 starters for a good test
    for _ in range(50):
        drop_organism(world, np.random.randint(W_GRID), np.random.randint(H_GRID))

    if is_headless:
        print(f"\nRunning simulation headless for {headless_ticks} ticks...")
        t0 = time.time()
        for i in range(headless_ticks):
            mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
            out = model.predict({"world": world, "mutation": mutation})
            world = list(out.values())[0]
            
            # Matched exactly to Pygame logic!
            world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
            world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
            
            if True:
                pop = (world[0, 1] > 0).sum()
                print(f"Tick {i:4d} | Population: {pop}")
        
        t1 = time.time()
        fps = headless_ticks / (t1 - t0)
        pop = (world[0, 1] > 0).sum()
        print(f"\nFinished. {headless_ticks} ticks in {t1-t0:.2f}s ({fps:.1f} ticks/sec). Final pop: {pop}")
        return

    running = True
    tick_count = 0
    print("\nSimulation Started!")
    print(" - Clicking spawns an organism.")
    
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
        mutation = (np.random.randn(1, CH_WEIGHTS, H_GRID, W_GRID) * 0.1).astype(np.float32)
        out = model.predict({"world": world, "mutation": mutation})
        world = list(out.values())[0]
        
        # Background physics: global food regeneration bounds
        world[0, 0] += np.random.rand(H_GRID, W_GRID) * 0.02
        world[0, 0] = np.clip(world[0,0], 0.0, 1.0)
        
        # Render visual channels
        t = world[0]
        
        # 1. Background Space & Food Surface
        screen.fill((15, 15, 20)) # Dark void
        rgba = np.zeros((H_GRID, W_GRID, 3), dtype=np.uint8)
        rgba[..., 1] = np.clip(t[0] * 70.0, 0, 255) # Dim green for food algae
        rgba = np.transpose(rgba, (1, 0, 2))
        surf = pygame.surfarray.make_surface(rgba)
        surf_scaled = pygame.transform.scale(surf, (W_PX, H_PX))
        screen.blit(surf_scaled, (0, 0))
        
        # 2. Draw Organisms dynamically
        orgs = t[1]
        weights = t[2:]
        y_idx, x_idx = np.nonzero(orgs > 0)
        pop = len(y_idx)
        pop_history.append(pop)
        
        for y, x in zip(y_idx, x_idx):
            energy = orgs[y, x]
            w = weights[:, y, x]
            
            # Lineage color projection from their 15 neural weights
            c = np.clip(np.dot(w, color_proj) * 80 + 128, 50, 255).astype(int)
            
            # Position & pulsating size based on energy
            cx = x * RENDER_SCALE + RENDER_SCALE // 2
            cy = y * RENDER_SCALE + RENDER_SCALE // 2
            r = int((0.5 + energy * 0.5) * RENDER_SCALE * 0.9)
            
            # Body & Core Outline
            pygame.draw.circle(screen, c, (cx, cy), r)
            pygame.draw.circle(screen, (255, 255, 255), (cx, cy), r, max(1, r//4))
            
            # Orientation heuristic: using their first trait [0] as an angular heading
            ax = cx + int(np.cos(w[0] * np.pi) * r * 1.8)
            ay = cy + int(np.sin(w[0] * np.pi) * r * 1.8)
            pygame.draw.line(screen, (255, 255, 255), (cx, cy), (ax, ay), max(1, RENDER_SCALE//8))
            
        # 3. Draw Side HUD
        hud_x = W_PX + 20
        # Title
        screen.blit(font_large.render("BEETLE-BRAIN", True, (200, 200, 220)), (hud_x, 20))
        screen.blit(font.render("DISCRETE ANE", True, (100, 100, 150)), (hud_x, 45))
        
        # Stats
        stats_y = 90
        screen.blit(font.render(f"tick {tick_count:7d}", True, (255, 255, 255)), (hud_x, stats_y)); stats_y += 20
        screen.blit(font.render(f"pop  {pop:7d}", True, (255, 255, 255)), (hud_x, stats_y)); stats_y += 40
        
        # Micro Line Chart for Population
        screen.blit(font.render("POPULATION HISTORY", True, (150, 150, 150)), (hud_x, stats_y)); stats_y += 20
        ch_w, ch_h = 280, 80
        pygame.draw.rect(screen, (25, 25, 30), (hud_x, stats_y, ch_w, ch_h))
        if len(pop_history) > 1:
            max_p = max(max(pop_history), 1)
            pts = [(hud_x + i, stats_y + ch_h - int((p / max_p) * ch_h)) for i, p in enumerate(pop_history)]
            pts.insert(0, (hud_x, stats_y + ch_h)) # Pin to bottom left
            pts.append((hud_x + len(pop_history) - 1, stats_y + ch_h)) # Pin to bottom right
            
            # Fill under the line chart
            pygame.draw.polygon(screen, (100, 60, 40), pts)
            # Top boundary line
            pts = [(hud_x + i, stats_y + ch_h - int((p / max_p) * ch_h)) for i, p in enumerate(pop_history)]
            pygame.draw.lines(screen, (240, 150, 50), False, pts, 2)
            
            # Live Max text 
            screen.blit(font.render(f"max: {max_p}", True, (100, 100, 100)), (hud_x + ch_w - 70, stats_y - 20))
            
        pygame.display.flip()
        
        clock.tick(60)
        tick_count += 1
        pygame.display.set_caption(f"Beetle Brain - ANE Discrete Evolution | {clock.get_fps():.0f} FPS")

    pygame.quit()

if __name__ == "__main__":
    main()