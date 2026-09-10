import pandas as pd, numpy as np
from sklearn.metrics import cohen_kappa_score

def interp(k):
    for t,l in [(0.81,'almost perfect'),(0.61,'substantial'),(0.41,'moderate'),(0.21,'fair'),(0.0,'slight')]:
        if k>=t: return l
    return 'poor'

mon = pd.read_csv('data/raw/signals_systems_validity_corpus.csv')
sam = pd.read_csv('data/raw/sam_marked.csv')

allm, alls = [], []
rows_out = []
print(f"{'CCU':<6}{'n':>5}{'agree':>9}{'kappa':>8}  interpretation")
print("-"*50)
for ccu in ['ccu1','ccu2','ccu3']:
    m = mon[mon['ccuname']==ccu].drop_duplicates('uid').set_index('uid')
    s = sam[sam['CCU']==ccu]
    M,S=[],[]
    for _, r in s.iterrows():
        uid = r['ID']
        if uid not in m.index: continue
        mv = 1 if m.loc[uid,'validity']=='correct' else 0
        sv = int(r['Free-Text Validity'])
        M.append(mv); S.append(sv)
        rows_out.append((uid, ccu, str(m.loc[uid,'q1txr'])[:80], mv, sv))
    k = cohen_kappa_score(M,S); agree = np.mean(np.array(M)==np.array(S))
    allm+=M; alls+=S
    print(f"{ccu:<6}{len(M):>5}{agree*100:>8.1f}%{k:>8.3f}  {interp(k)}")

k_all = cohen_kappa_score(allm,alls); a_all=np.mean(np.array(allm)==np.array(alls))
print("-"*50)
print(f"{'ALL':<6}{len(allm):>5}{a_all*100:>8.1f}%{k_all:>8.3f}  {interp(k_all)}")

import csv
with open('outputs/agreement/disagreement_rows.csv','w',newline='') as f:
    w = csv.writer(f); w.writerow(['uid','ccu','text','monica_validity','sam_validity'])
    for row in rows_out:
        if row[3] != row[4]:
            w.writerow(row)
print(f"\nDisagreements written to outputs/agreement/disagreement_rows.csv")
