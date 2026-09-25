
import numpy as np

def require_complete_assignment(required,assigned):
    if (len(required)!=len(set(required)) or len(assigned)!=len(set(assigned))
            or set(required)!=set(assigned)):
        raise ValueError('assignment must cover the complete required atom inventory once')

def score_roles(roles, coordinates_angstrom, atom_mask, residue_types, tolerance=1.5):
    xyz=np.asarray(coordinates_angstrom,dtype=float)
    mask=np.asarray(atom_mask,dtype=bool)
    rt=np.asarray(residue_types)
    if not roles or xyz.shape != mask.shape+(3,) or len(rt)!=len(mask):
        raise ValueError('incomplete role or coordinate inventory')
    atoms=[]
    for role in roles:
        r,a=role['residue'],role['atom']
        present=bool(mask[r,a]);identity=bool(rt[r]==role['residue_type'])
        distance=None
        if present:
            distance=float(np.linalg.norm(xyz[r,a]-role['xyz_angstrom']))
            if not np.isfinite(distance):
                raise ValueError('nonfinite present atom')
        placed=present and distance<=tolerance
        atoms.append({**role,'present':present,'identity_match':identity,
                      'displacement_angstrom':distance,'geometrically_placed':placed,
                      'identity_and_placed':bool(identity and placed)})
    return {'required':len(atoms),'present':sum(a['present'] for a in atoms),
            'geometrically_placed':sum(a['geometrically_placed'] for a in atoms),
            'identity_and_placed':sum(a['identity_and_placed'] for a in atoms),
            'all_identity_and_placed':all(a['identity_and_placed'] for a in atoms),
            'all_identity':all(a['identity_match'] for a in atoms),'atoms':atoms}
