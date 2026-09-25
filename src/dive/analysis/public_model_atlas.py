
from collections import Counter, defaultdict
import re
import numpy as np

def conjunction(*values):
    if any(v is False for v in values):
        return False
    return True if all(v is True for v in values) else None

def cross_table(rows):
    patterns = Counter()
    eligible = failure = missing = success = 0
    for r in rows:
        patterns['/'.join('P' if r.get(k) is True else 'F' if r.get(k) is False
                          else 'U' for k in ('G', 'S', 'L'))] += 1
        if conjunction(r.get('G'), r.get('S')) is True:
            eligible += 1
            failure += r.get('L') is False
            missing += r.get('L') is None
        success += conjunction(r.get('G'), r.get('S'), r.get('L')) is True
    return dict(total=len(rows), patterns=dict(sorted(patterns.items())),
                eligible_GS=eligible, local_failure=failure,
                local_unmeasured_among_GS=missing, joint_success=success,
                local_failure_fraction=(failure / eligible if eligible and not missing
                                        else None))

def paired_effect(rows, arm_a, arm_b, metric, n_boot=10000):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        value = r.get(metric)
        if value is not None and np.isfinite(value):
            groups[r['parent_id']][r['arm']].append(float(value))
    pairs = [{'parent_id': p, 'a': float(np.mean(g[arm_a])),
              'b': float(np.mean(g[arm_b])),
              'n_a': len(g[arm_a]), 'n_b': len(g[arm_b])}
             for p, g in sorted(groups.items()) if arm_a in g and arm_b in g]
    for p in pairs:
        p['delta'] = p['a'] - p['b']
    d = np.array([p['delta'] for p in pairs])
    out = dict(arm_a=arm_a, arm_b=arm_b, metric=metric, pairs=pairs,
               n_parents=len(d), mean_delta=float(d.mean()) if len(d) else None,
               ci95=None, leave_one_parent_out=[],uncertainty_status='insufficient parents')
    if len(d)>=2 and np.ptp(d)==0:
        out['uncertainty_status']='constant empirical effects; bootstrap is uninformative'
    elif len(d) >= 2:
        boot = np.random.default_rng(20260906).choice(d, (n_boot,len(d))).mean(1)
        out['ci95'] = np.quantile(boot, [.025,.975]).tolist()
        out['leave_one_parent_out'] = [float(np.delete(d,i).mean()) for i in range(len(d))]
        out['uncertainty_status']='exploratory parent bootstrap; no multiplicity correction'
    return out

def bond_geometry(residues):
    nums = sorted(residues)
    pairs = [(a,b) for a,b in zip(nums,nums[1:]) if b == a+1]
    out = dict(n_residues=len(nums), adjacent_pairs=len(pairs),
               numbering_gaps=sum(b != a+1 for a,b in zip(nums,nums[1:])))
    gates = []
    for name, aa, bb, lo, hi in [('C_N','C','N',1.2,1.45),
                                 ('CA_CA','CA','CA',3.5,4.2)]:
        vals = [float(np.linalg.norm(np.asarray(residues[a][aa])-residues[b][bb]))
                for a,b in pairs if aa in residues[a] and bb in residues[b]]
        finite = all(np.isfinite(vals))
        fraction = sum(lo < v < hi for v in vals)/len(pairs) if pairs else None
        missing = len(pairs)-len(vals)
        gate = (bool(fraction >= .95) if pairs and not missing and finite else None)
        out[name] = dict(n=len(vals), missing=missing, fraction=fraction,
                         mean=float(np.mean(vals)) if vals and finite else None,
                         valid=gate)
        gates.append(gate)
    out['valid'] = conjunction(*gates)
    return out

def fitted_rmsd(fit_mobile, fit_target, score_mobile=None, score_target=None):
    a,b = np.asarray(fit_mobile,float),np.asarray(fit_target,float)
    if a.shape != b.shape or len(a)<3 or a.ndim!=2 or a.shape[1]!=3:
        raise ValueError('fit coverage/shape')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('nonfinite fit')
    ac,bc=a.mean(0),b.mean(0)
    u,_,vt=np.linalg.svd((a-ac).T@(b-bc))
    rot=u@np.diag([1,1,np.linalg.det(u@vt)])@vt
    x,y=(a,b) if score_mobile is None else (np.asarray(score_mobile),np.asarray(score_target))
    if x.shape!=y.shape or not len(x):
        raise ValueError('score coverage/shape')
    return float(np.sqrt(np.mean(np.sum(((x-ac)@rot+bc-y)**2,axis=1))))

def sequence_by_number(numbers, sequence):
    if len(numbers)!=len(sequence) or len(set(numbers))!=len(numbers):
        raise ValueError('sequence length or duplicate numbering')
    return dict(zip(numbers, sequence))

def region_numbers(numbers, selector, chain):
    m=re.fullmatch(r'([^:]+):(\d+)-(\d+)', selector)
    if not m:
        raise ValueError('region must be an input chain:lo-hi selector')
    return [n for n in numbers if chain==m[1] and int(m[2])<=n<=int(m[3])]
