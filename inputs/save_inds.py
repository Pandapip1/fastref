import numpy as np
import mdtraj as md

sim_name = 'TODO_FILLIN_inputs_save_inds-nohydrogen'

pdb = md.load(f"{sim_name}-prot-masses.pdb")
atom_inds = pdb.topology.select(f"protein and name N CA CB C O")

for i in atom_inds:
    print(i)
