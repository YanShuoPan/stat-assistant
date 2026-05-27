"""
論文搜尋工具
用法: python tools/fetch_paper.py "論文標題"

透過 Google Scholar 搜尋論文，列出結果後：
- 免費 PDF → 直接下載
- 付費期刊 → 用瀏覽器打開（透過學校 VPN 存取）
"""

import sys
import io
import re
import webbrowser
import requests
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

from scholarly import scholarly

DOWNLOAD_DIR = Path(__file__).parent.parent / "papers"
DOWNLOAD_DIR.mkdir(exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def sanitize_filename(title: str) -> str:
    name = re.sub(r'[<>:"/\|?*]', '', title)
    return name.strip()[:120] + ".pdf"


def try_direct_download(url: str, filepath: Path) -> bool:
    """嘗試直接下載 PDF，成功回傳 True"""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, allow_redirects=True)
        resp.raise_for_status()
        if b'%PDF' in resp.content[:20]:
            filepath.write_bytes(resp.content)
            return True
    except requests.RequestException:
        pass
    return False


def search_and_download(query: str) -> None:
    print(f"\n搜尋: {query}")
    print("-" * 60)

    results = scholarly.search_pubs(query)
    papers = []

    for i, paper in enumerate(results):
        if i >= 5:
            break
        papers.append(paper)

        title = paper['bib'].get('title', '(無標題)')
        author = ', '.join(paper['bib'].get('author', []))[:80]
        year = paper['bib'].get('pub_year', '?')
        venue = paper['bib'].get('venue', '')
        eprint = paper.get('eprint_url', '')
        pub_url = paper.get('pub_url', '')

        print(f"\n[{i+1}] {title}")
        print(f"    作者: {author}")
        print(f"    年份: {year}  期刊: {venue}")
        if eprint:
            print(f"    PDF: {eprint}")
        if pub_url:
            print(f"    連結: {pub_url}")

    if not papers:
        print("找不到結果")
        return

    print("\n" + "-" * 60)
    choice = input("輸入編號開啟/下載 (0=取消, r=重新搜尋): ").strip()

    if choice == '0' or choice == '':
        print("已取消")
        return
    if choice == 'r':
        new_query = input("輸入新的搜尋關鍵字: ").strip()
        if new_query:
            search_and_download(new_query)
        return

    try:
        idx = int(choice) - 1
    except ValueError:
        print("無效輸入")
        return

    if idx < 0 or idx >= len(papers):
        print("編號超出範圍")
        return

    selected = papers[idx]
    title = selected['bib'].get('title', 'paper')
    eprint = selected.get('eprint_url', '')
    pub_url = selected.get('pub_url', '')
    filepath = DOWNLOAD_DIR / sanitize_filename(title)

    # 1) 先嘗試直接下載免費 PDF
    if eprint:
        print(f"\n嘗試直接下載: {eprint}")
        if try_direct_download(eprint, filepath):
            print(f"下載成功! 已儲存: {filepath}")
            return
        print("直接下載失敗（需要權限）")

    # 2) 無法直接下載 → 用瀏覽器打開（走 VPN）
    open_url = eprint or pub_url
    if open_url:
        print(f"\n用瀏覽器開啟: {open_url}")
        print("(請確認已連上學校 VPN)")
        webbrowser.open(open_url)
    else:
        print("找不到可用的連結")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("輸入論文標題或關鍵字: ").strip()

    if not query:
        print("請提供搜尋關鍵字")
        sys.exit(1)

    search_and_download(query)
