"""Descriptive binary classification utilities for optional diagnostics.

An interference threshold is not a theorem certifying safe SF composition.
It is excluded from the repaired method and its evaluation protocol.
"""
import numpy as np

def relative_gap(retrain_value, transfer_value):
    if not np.isfinite([retrain_value,transfer_value]).all():raise ValueError('values must be finite')
    denom=abs(float(retrain_value))
    return float((retrain_value-transfer_value)/denom) if denom else (0. if transfer_value==0 else float('inf'))

def oracle_paid(rel_gap,tol=.05):return bool(rel_gap>tol)
def gate_paid(rate,threshold):return bool(rate>threshold)
def evaluate_gate(tasks,threshold,tol=.05):
    tp=fp=tn=fn=0
    for row in tasks:
        truth=oracle_paid(row['rel_gap'],tol); pred=gate_paid(row['rate'],threshold)
        tp+=bool(truth and pred);fp+=bool(not truth and pred)
        tn+=bool(not truth and not pred);fn+=bool(truth and not pred)
    n=tp+fp+tn+fn;p=tp/(tp+fp) if tp+fp else float('nan');r=tp/(tp+fn) if tp+fn else float('nan')
    return {'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':p,'recall':r,
            'f1':2*p*r/(p+r) if np.isfinite(p+r) and p+r else float('nan'),
            'accuracy':(tp+tn)/n if n else float('nan'),
            'savings':(tn+fn)/n if n else float('nan')}
def threshold_sweep(tasks,grid=41,tol=.05):
    if not tasks:raise ValueError('empty diagnostic tasks')
    rates=[float(t['rate']) for t in tasks]
    if not np.isfinite(rates).all() or grid<2:raise ValueError('invalid diagnostic data')
    curve=[(float(t),evaluate_gate(tasks,float(t),tol)) for t in np.linspace(0,max(rates)*1.05+1e-6,grid)]
    return max(curve,key=lambda c:c[1]['f1'] if np.isfinite(c[1]['f1']) else -1),curve
