#!/usr/bin/env python
import sys

def remove_hydrogens(input_pdb, output_pdb):
    with open(input_pdb, 'r') as infile, open(output_pdb, 'w') as outfile:
        for line in infile:
            if line.startswith(("ATOM", "HETATM")):
                atom_name = line[12:16].strip()
                if not atom_name.startswith('H'):
                    outfile.write(line)
            else:
                outfile.write(line)

if __name__ == "__main__":
    remove_hydrogens(sys.argv[1], sys.argv[2])
