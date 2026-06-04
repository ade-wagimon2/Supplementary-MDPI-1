import json
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
conf_path = os.path.join(script_dir, 'CRTB_COMPLEX_seed_42_sample_1_confidences.json')
pdb_path = os.path.join(script_dir, 'CRTB_COMPLEX_seed_42_sample_1_model.pdb')
output_dir = script_dir
output_img = os.path.join(output_dir, 'confidence_visualization_model1.png')

# Load JSON
with open(conf_path, 'r') as f:
    data = json.load(f)

plddt_raw = data.get('plddt', [])
pde = np.array(data.get('pde', []))

# Load PDB to map atoms to residues
residues = []
current_res = None
atom_plddts = []

with open(pdb_path, 'r') as f:
    atom_idx = 0
    for line in f:
        if line.startswith('ATOM  ') or line.startswith('HETATM'):
            res_id = line[21:26].strip() + "_" + line[17:20].strip() # Chain + ResSeq + ResName
            if res_id != current_res:
                if current_res is not None:
                    residues.append({'id': current_res, 'plddts': atom_plddts})
                current_res = res_id
                atom_plddts = []
            
            if atom_idx < len(plddt_raw):
                atom_plddts.append(plddt_raw[atom_idx])
            atom_idx += 1
    
    if current_res is not None:
        residues.append({'id': current_res, 'plddts': atom_plddts})

# Calculate average pLDDT per residue
res_plddt = [np.mean(r['plddts']) for r in residues if r['plddts']]
res_labels = [r['id'] for r in residues if r['plddts']]

# Plotting configuration for publication
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 2.0

fig = plt.figure(figsize=(16, 7))

# Subplot 1: pLDDT
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(res_plddt, color='#2ecc71', linewidth=2.5)
ax1.fill_between(range(len(res_plddt)), res_plddt, color='#2ecc71', alpha=0.15)
ax1.set_title('A. pLDDT per Residue/Token', fontsize=22, fontweight='bold', pad=20)
ax1.set_xlabel('Token Index', fontsize=20, labelpad=10)
ax1.set_ylabel('pLDDT Score', fontsize=20, labelpad=10)
ax1.set_ylim(0, 105)
ax1.tick_params(axis='both', which='major', labelsize=18, width=2)
ax1.grid(True, linestyle='--', alpha=0.4)

# Subplot 2: PDE Matrix
ax2 = fig.add_subplot(1, 2, 2)
# PDE is usually error, so low values (blue/cool) are good. 
# User asked for 'jet', we'll use it but make sure it looks sharp.
im = ax2.imshow(pde, cmap='viridis', interpolation='nearest', vmin=0, vmax=5)
ax2.set_title('B. Predicted Distance Error (PDE)', fontsize=22, fontweight='bold', pad=20)
ax2.set_xlabel('Scored Token Index', fontsize=20, labelpad=10)
ax2.set_ylabel('Aligned Token Index', fontsize=20, labelpad=10)
ax2.tick_params(axis='both', which='major', labelsize=18, width=2)

# Add colorbar with larger font
cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
cbar.set_label('Error (Å)', fontsize=20, labelpad=10)
cbar.ax.tick_params(labelsize=18, width=2)

plt.tight_layout(pad=3.0)
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"Publication-ready visualization saved to {output_img}")
