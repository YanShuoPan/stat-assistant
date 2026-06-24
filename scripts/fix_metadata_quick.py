#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'apps', 'api'))
os.environ.setdefault('DATABASE_URL', 'sqlite:///apps/api/dev.db')
os.environ.setdefault('JWT_SECRET_KEY', 'unused')
os.environ.setdefault('OPENAI_API_KEY', 'unused')

from database import SessionLocal
from models import Paper

db = SessionLocal()

updates = [
    (1, 'Victor Chernozhukov, Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, Whitney Newey, James Robins', 2018, '10.1111/ectj.12097'),
    (2, 'Ching-Kang Ing, Tze Leung Lai', 2011, None),
    (5, 'Yuxiao Chen, Yun Li, Tao Li, Christa Schlegel, Noel Cressie', 2024, '10.1080/10618600.2023.2257239'),
    (71, None, 2025, '10.1007/s11222-025-10775-8'),
    (82, None, 2025, '10.1111/sjos.70006'),
    (150, 'T. Tony Cai, Hongji Wei', 2022, None),
    (183, None, 2025, '10.1093/biomtc/ujaf074'),
    (184, None, 2024, '10.1214/24-ejs2295'),
]

for paper_id, authors, year, doi in updates:
    p = db.query(Paper).filter(Paper.id == paper_id).first()
    if not p:
        print(f'ID={paper_id} NOT FOUND')
        continue
    changed = []
    if authors is not None:
        if p.authors != authors:
            p.authors = authors
            changed.append(f'authors={authors[:50]}')
    else:
        if p.authors in ('Unknown (PDF unreadable)',):
            p.authors = None
            changed.append('authors=NULL')
    if year and p.year != year:
        p.year = year
        changed.append(f'year={year}')
    if doi and not p.doi:
        p.doi = doi
        changed.append(f'doi={doi}')
    if changed:
        db.add(p)
        print(f'ID={paper_id}: {" | ".join(changed)}')
    else:
        print(f'ID={paper_id}: no change')

db.commit()
print('Done.')
db.close()
