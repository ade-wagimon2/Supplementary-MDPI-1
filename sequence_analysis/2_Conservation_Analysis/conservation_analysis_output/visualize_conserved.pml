# PyMOL Visualization Script
# Generated for: 3W7F_A.pdb
# Auto-generated - do not edit manually

# Reinitialize PyMOL
reinitialize

# Load structure
load ../3W7F_A.pdb, 3W7F_A

# Set visualization style
hide all
show cartoon, 3W7F_A
color gray80, 3W7F_A

# Create selections and color residue groups

# conserved_same (20 residues)
select conserved_same, 3W7F_A and resi 41+45+48+52+114+129+134+138+160+161+163+165+168+171+172+176+181+183+185+248
color green, conserved_same
show sticks, conserved_same

# unique_psy (5 residues)
select unique_psy, 3W7F_A and resi 34+157+229+233+265
color marine, unique_psy
show sticks, unique_psy

# unique_crtm (43 residues)
select unique_crtm, 3W7F_A and resi 11+19+22+25+26+29+30+36+37+39+43+44+49+65+87+89+90+104+110+111+131+133+137+140+141+143+164+167+169+170+174+175+208+212+213+216+219+223+251+252+256+260+268
color salmon, unique_crtm
show sticks, unique_crtm

# Final view settings
deselect
orient
zoom vis
bg_color white
set ray_trace_mode, 1

# Save session (optional)
# save visualize_conserved.pse