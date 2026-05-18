from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # D:/aic

trans = set()
for vd in (BASE_DIR / 'dataset' / 'transcripts').iterdir():
    if vd.is_dir():
        for f in vd.glob('*.json'):
            trans.add((vd.name, f.stem))

feats = set()
for vd in (BASE_DIR / 'data' / 'features' / 'bgem3').iterdir():
    if vd.is_dir():
        for f in vd.glob('*_tokens.npz'):
            shot_id = f.stem.replace('_tokens', '')
            feats.add((vd.name, shot_id))

missing = trans - feats
print(f'Missing: {len(missing)} shots')
for vid, shot in sorted(missing):
    print(f' {vid}/{shot}.json')