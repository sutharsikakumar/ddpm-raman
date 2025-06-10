from ase.io import read, write
cell = read('scf-supercell-10x20.in', format='espresso-in')
print("Atoms in cell:", len(cell))
write('data.mos2', cell, format='lammps-data', specorder=['Mo','S'])
print("Wrote data.mos2")