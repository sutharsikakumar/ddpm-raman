from ase.build import make_supercell
from ase.io import read, write

atoms = read('mos2_pristine.in', format='espresso-in')
supercell = make_supercell(atoms, [[5, 0, 0], [0, 5, 0], [0, 0, 1]])
write('mos2_pristine_5x5.in', supercell, format='espresso-in')