#!/usr/bin/env python
import numpy as np
import mdtraj as md
import sys

if __name__ == "__main__":
	pdb = md.load(sys.argv[1])
	atom_inds = pdb.topology.select(f"protein and name N CA CB C O")
	with open(sys.argv[2], "w") as f:
		for i in atom_inds:
			f.write(f"${i}\n")
