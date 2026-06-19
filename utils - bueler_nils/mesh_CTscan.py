import trimesh
import numpy as np
import gmsh
import os
import math

# --- Configuration ---
INPUT_STL = r"C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Feb26/CT_Teil1288_Zugprobe1_5k_2.stl"       # Replace with your actual file name
OUTPUT_INP = r"C:\Users\KoltzenburgNils\OneDrive - inspire AG\Diss\93_Buehler\Versuche_Feb26/homogenized_mesh5.inp"
LOCAL_MESH_SIZE = 0.7             # Fine mesh size at the pores
GLOBAL_MESH_SIZE = 1.5            # Coarse mesh size for the bulk matrix
WARNING_THRESHOLD = 1.5 * LOCAL_MESH_SIZE 

def process_geometry_and_mesh():
    print("1. Loading and splitting the STL...")
    # Load the monolithic STL
    scene = trimesh.load(INPUT_STL, force='mesh')
    
    # Split into separate watertight bodies
    components = scene.split(only_watertight=False)
    
    # Sort by volume: largest is the outer box, the rest are pores
    components.sort(key=lambda m: m.volume, reverse=True)
    raw_box = components[0]
    pores = components[1:]
    
    print(f"   Found 1 outer boundary and {len(pores)} internal pores.")

    print("\n2. Calculating Oriented Bounding Box (OBB) & Aligning...")
    # Get the OBB of the raw box to handle the tilt
    obb_transform = raw_box.bounding_box_oriented.primitive.transform
    extents = raw_box.bounding_box_oriented.primitive.extents
    
    # Inverse transform to align everything to the global axes (centered at origin)
    align_matrix = np.linalg.inv(obb_transform)
    
    # Apply alignment to all pores
    for pore in pores:
        pore.apply_transform(align_matrix)
        
    print(f"   Aligned box dimensions (LxWxH): {extents[0]:.2f} x {extents[1]:.2f} x {extents[2]:.2f}")

    print("\n3. Checking pore proximity to analytical boundaries...")
    # Box limits (since it's centered at the origin)
    half_extents = extents / 2.0
    
    valid_pores = []
    pore_stls = []
    for i, pore in enumerate(pores):
        # Find the minimum distance from pore vertices to the box boundaries
        vertices = pore.vertices
        dist_x = half_extents[0] - np.abs(vertices[:, 0])
        dist_y = half_extents[1] - np.abs(vertices[:, 1])
        dist_z = half_extents[2] - np.abs(vertices[:, 2])
        
        min_dist = np.min([dist_x, dist_y, dist_z])
        
        if min_dist < 0:
            print(f"   [ERROR] Pore {i} intersects the boundary! Skipping to prevent mesh failure.")
            continue
        elif min_dist < WARNING_THRESHOLD:
            print(f"   [WARNING] Pore {i} is dangerously close to the wall (Distance: {min_dist:.3f}). Watch for distorted C3D10 elements.")
        
        valid_pores.append(pore)
        fname = f"temp_pore_{i}.stl"
        pore.export(fname)
        pore_stls.append(fname)

    print("\n4. Meshing with Gmsh (Native CAD + Discrete Voids)...")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("Homogenization_Model")
    
    surface_loops = []
    
    # --- A. Create Native CAD Box ---
    # Centered at origin to match the aligned pores
    gmsh.model.occ.addBox(-half_extents[0], -half_extents[1], -half_extents[2], 
                          extents[0], extents[1], extents[2], 1)
    gmsh.model.occ.synchronize()
    
    # TRICK: Delete the OCC Volume (3D entity 1) but KEEP the 6 surfaces (recursive=False).
    # This allows us to redefine the volume later with the discrete STL holes included.
    gmsh.model.removeEntities([(3, 1)], recursive=False)
    
    # Grab the 6 perfect CAD surfaces and make the outer loop
    box_surfs = [e[1] for e in gmsh.model.getEntities(2)]
    outer_loop = gmsh.model.geo.addSurfaceLoop(box_surfs)
    surface_loops.append(outer_loop)
    
    # --- B. Load Pores (Discrete STLs) ---
    pore_surf_ids = []
    for p_file in pore_stls:
        entities_before = gmsh.model.getEntities(2)
        gmsh.merge(p_file)
        entities_after = gmsh.model.getEntities(2)
        
        # Identify the surfaces that were just loaded
        current_pore_surfs = [e[1] for e in entities_after if e not in entities_before]
        pore_surf_ids.extend(current_pore_surfs)
        
        inner_loop = gmsh.model.geo.addSurfaceLoop(current_pore_surfs)
        surface_loops.append(inner_loop)
        
    # --- C. Define the "Swiss Cheese" Volume ---
    volume_id = gmsh.model.geo.addVolume(surface_loops)
    gmsh.model.geo.synchronize()

    # --- D. Physical Groups (The Abaqus Filter) ---
    # By creating a Physical Group for the Volume, Gmsh knows to ONLY export the 3D elements
    # to the .inp file, automatically stripping out the 1D lines and 2D triangles.
    gmsh.model.addPhysicalGroup(3, [volume_id], name="Matrix_Volume")
    gmsh.option.setNumber("Mesh.SaveAll", 0) # Strictly enforce saving only physical groups

    # --- E. Dual Constraint Mesh Sizing ---
    
    # 1. The Ceiling: Cap the absolute maximum element size globally (This fixes the box walls)
    gmsh.option.setNumber("Mesh.MeshSizeMax", GLOBAL_MESH_SIZE)
    gmsh.option.setNumber("Mesh.MeshSizeMin", LOCAL_MESH_SIZE)
    
    # 2. The Source: Calculate distance ONLY from the pores
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "SurfacesList", pore_surf_ids)
    
    # 3. The Growth: Transition smoothly from pores outward
    gmsh.model.mesh.field.add("MathEval", 2)
    # Starts at LOCAL_MESH_SIZE and grows by 0.2 units per unit of distance
    eval_string = f"{LOCAL_MESH_SIZE} + F1 * 0.2"
    gmsh.model.mesh.field.setString(2, "F", eval_string)
    
    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    # Generate 3D Mesh
    gmsh.model.mesh.generate(3)
    
    # Convert to Second Order (C3D10)
    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0) 
    gmsh.model.mesh.setOrder(2)
    
    print("\n5. Exporting to Abaqus .inp...")
    gmsh.write(OUTPUT_INP)
    gmsh.finalize()
    
    # Clean up temp files
    for p_file in pore_stls:
        if os.path.exists(p_file):
            os.remove(p_file)
        
    print(f"\nSuccess! Mesh exported to {OUTPUT_INP}")

if __name__ == "__main__":
    process_geometry_and_mesh()