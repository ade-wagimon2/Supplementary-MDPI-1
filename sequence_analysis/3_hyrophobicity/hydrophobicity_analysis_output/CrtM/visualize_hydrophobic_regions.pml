# PyMOL Hydrophobic Regions Visualization Script
# Generated for: 3W7F_A.pdb

reinitialize

load ../../3W7F_A.pdb, 3W7F_A

# Base visualization
hide all
show cartoon, 3W7F_A
color gray80, 3W7F_A

# Highlight 2 hydrophobic regions

# Region 1: residues 133-148 (mean hydro: 1.29)
select hydro_region_1, 3W7F_A and resi 133+134+135+136+137+138+139+140+141+142+143+144+145+146+147+148
color firebrick, hydro_region_1
show surface, hydro_region_1
set transparency, 0.5, hydro_region_1

# Region 2: residues 159-166 (mean hydro: 0.75)
select hydro_region_2, 3W7F_A and resi 159+160+161+162+163+164+165+166
color orange, hydro_region_2
show surface, hydro_region_2
set transparency, 0.5, hydro_region_2

# Final settings
deselect
orient
zoom vis
bg_color white