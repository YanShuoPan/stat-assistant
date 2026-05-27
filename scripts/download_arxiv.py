"""Download arXiv/OA PDFs via Semantic Scholar DOI lookup — all domains.

Usage:
    python scripts/download_arxiv.py                      # run all domains
    python scripts/download_arxiv.py --domain bayesian    # substring match
    python scripts/download_arxiv.py --domain hd          # matches high-dimensional
    python scripts/download_arxiv.py --cluster 0          # only cluster_0
    python scripts/download_arxiv.py --dry-run            # search only
    python scripts/download_arxiv.py --limit 50           # cap per cluster
    python scripts/download_arxiv.py --list               # show available domains

Resumes automatically. Uses Semantic Scholar DOI lookup.
"""

import argparse, json, re, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent / "papers"
CONFIG_FILE = BASE_DIR / "download_config.json"
PROGRESS_FILE = BASE_DIR / "arxiv_download_progress.json"
S2_API = "https://api.semanticscholar.org/graph/v1/paper"

SEARCH_DELAY = 4.0
DL_DELAY = 2.0
MAX_RETRIES = 3


def sanitize_filename(title):
    name = re.sub(r'[^\w\s\-]', '', title)
    name = re.sub(r'\s+', ' ', name).strip()
    return name[:120]


def lookup_by_doi(doi):
    if doi.startswith('http'):
        doi = doi.replace('https://doi.org/', '').replace('http://doi.org/', '')
    url = f'{S2_API}/DOI:{urllib.parse.quote(doi, safe="")}?fields=title,externalIds,openAccessPdf'
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'ModelBridge/1.0'})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            ext = data.get('externalIds', {})
            arxiv_id = ext.get('ArXiv')
            oa = data.get('openAccessPdf')
            pdf_url = oa['url'] if oa else None
            if arxiv_id and not pdf_url:
                pdf_url = f'https://arxiv.org/pdf/{arxiv_id}'
            if arxiv_id or pdf_url:
                return {'arxiv_id': arxiv_id or '', 'pdf_url': pdf_url or '', 's2_title': data.get('title', '')}
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = SEARCH_DELAY * (2 ** attempt) + 2
                print(f'    [429] Rate limited, waiting {wait:.0f}s...')
                time.sleep(wait)
            elif e.code == 404:
                return None
            else:
                print(f'    [ERROR] HTTP {e.code}')
                return None
        except Exception as e:
            print(f'    [ERROR] {e}')
            return None
    return None


def download_pdf(pdf_url, save_path):
    for attempt in range(2):
        try:
            req = urllib.request.Request(pdf_url, headers={'User-Agent': 'ModelBridge/1.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if not data[:5].startswith(b'%PDF') and len(data) < 1000:
                print(f'    [WARN] Not a PDF ({len(data)} bytes)')
                return False
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_path.write_bytes(data)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(5 * (attempt + 1))
            else:
                print(f'    [ERROR] Download HTTP {e.code}')
                return False
        except Exception as e:
            print(f'    [ERROR] Download: {e}')
            return False
    return False


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text(encoding='utf-8'))
    return {'searched': {}, 'found': 0, 'not_found': 0, 'downloaded': 0, 'no_doi': 0}


def save_progress(progress):
    PROGRESS_FILE.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')


def match_domains(config, pattern):
    """Return list of domain keys matching a substring pattern."""
    pattern = pattern.lower().replace('-', '_').replace(' ', '_')
    matched = [k for k in config if pattern in k.lower()]
    if not matched:
        matched = [k for k in config if pattern in config[k]['pdf_parent'].lower()]
    return sorted(matched)


def main():
    parser = argparse.ArgumentParser(description='Download arXiv/OA PDFs for all domains')
    parser.add_argument('--domain', type=str, default=None, help='Domain substring filter (e.g. bayesian, hd)')
    parser.add_argument('--cluster', type=int, default=None, help='Only process this cluster index')
    parser.add_argument('--dry-run', action='store_true', help='Search only, no download')
    parser.add_argument('--limit', type=int, default=None, help='Max papers per cluster')
    parser.add_argument('--list', action='store_true', help='List available domains and exit')
    args = parser.parse_args()

    config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))

    if args.list:
        print(f'Available domains ({len(config)}):')
        for k, v in sorted(config.items()):
            print(f"  {k:40s} {v['total_papers']:>5d} papers, {v['num_clusters']} clusters -> {v['pdf_parent']}")
        return

    # Select domains
    if args.domain:
        domain_keys = match_domains(config, args.domain)
        if not domain_keys:
            print(f"No domain matching '{args.domain}'. Use --list to see available domains.")
            return
        print(f"Matched domains: {', '.join(domain_keys)}")
    else:
        domain_keys = sorted(config.keys())

    progress = load_progress()
    searched = progress['searched']

    grand_found = 0
    grand_searched = 0
    grand_downloaded = 0

    all_dir = BASE_DIR / 'pdf' / 'all'
    all_dir.mkdir(parents=True, exist_ok=True)

    for domain_key in domain_keys:
        dc = config[domain_key]
        clusters_file = BASE_DIR / dc['clusters_file']
        cluster_folders = dc['cluster_folders']

        if not clusters_file.exists():
            print(f"[SKIP] {domain_key}: {dc['clusters_file']} not found")
            continue

        raw = json.loads(clusters_file.read_text(encoding='utf-8'))
        clusters = raw.get('clusters', raw)  # support merged format

        cluster_keys = sorted(clusters.keys())
        if args.cluster is not None:
            cluster_keys = [f'cluster_{args.cluster}']

        dsep = '#' * 60
        print(f'\n{dsep}')
        print(f"DOMAIN: {domain_key} ({dc['total_papers']} papers, {len(cluster_keys)} clusters)")
        print(dsep)

        for ck in cluster_keys:
            if ck not in clusters:
                print(f'  Cluster {ck} not found in {domain_key}')
                continue
            idx = ck.split('_')[1]
            folder_name = cluster_folders.get(idx, f'cluster_{idx}')

            papers = clusters[ck]['papers']
            sep = '=' * 60
            print(f'\n{sep}')
            print(f'  {ck} -> {folder_name} ({len(papers)} papers)')
            print(sep)

            count = 0
            for i, paper in enumerate(papers):
                if args.limit and count >= args.limit:
                    break
                title = paper['title']
                doi = paper.get('doi', '')

                if title in searched:
                    if searched[title] not in ('not_found', 'no_doi'):
                        grand_found += 1
                    grand_searched += 1
                    continue

                if not doi:
                    searched[title] = 'no_doi'
                    progress['no_doi'] = progress.get('no_doi', 0) + 1
                    continue

                count += 1
                grand_searched += 1
                safe_title = title.encode('ascii', 'replace').decode()
                print(f'  [{i+1}/{len(papers)}] {safe_title[:70]}...')

                result = lookup_by_doi(doi)
                time.sleep(SEARCH_DELAY)

                if result is None:
                    searched[title] = 'not_found'
                    progress['not_found'] = progress.get('not_found', 0) + 1
                    print('    -> No arXiv/OA version')
                else:
                    grand_found += 1
                    arxiv_id = result['arxiv_id']
                    searched[title] = arxiv_id or 'oa_only'
                    progress['found'] = progress.get('found', 0) + 1
                    label = f'arXiv:{arxiv_id}' if arxiv_id else 'OA'
                    print(f'    -> FOUND ({label})')

                    if not args.dry_run and result['pdf_url']:
                        filename = sanitize_filename(title) + '.pdf'
                        save_path = all_dir / filename
                        if save_path.exists():
                            print('    -> Already downloaded')
                        else:
                            ok = download_pdf(result['pdf_url'], save_path)
                            if ok:
                                grand_downloaded += 1
                                progress['downloaded'] = progress.get('downloaded', 0) + 1
                                size_mb = save_path.stat().st_size / 1024 / 1024
                                print(f'    -> Downloaded ({size_mb:.1f} MB)')
                            time.sleep(DL_DELAY)

                if count % 10 == 0:
                    save_progress(progress)

            save_progress(progress)

    sep = '=' * 60
    print(f'\n{sep}')
    print('Summary:')
    print(f'  This run: searched={grand_searched} found={grand_found} downloaded={grand_downloaded}')
    sr = len(searched)
    fd = progress.get("found", 0)
    dl = progress.get("downloaded", 0)
    nd = progress.get("no_doi", 0)
    nf = progress.get("not_found", 0)
    print(f'  All-time: searched={sr} found={fd} downloaded={dl}')
    print(f'  No DOI: {nd} | Not on arXiv: {nf}')
    print(sep)


if __name__ == '__main__':
    main()
