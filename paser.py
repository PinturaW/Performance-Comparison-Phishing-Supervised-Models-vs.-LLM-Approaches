# read_paste_and_analyze.py
# -----------------------------------------------------------------------------
# ใช้ URL จาก paste-2.txt (black) / paste-3.txt (white) เป็นแหล่งจริง
# อ่าน Domain/Domain Age จาก URL (WHOIS; ไม่เจอ/0/error => 'suspect')
# วิเคราะห์ Title/Meta/Scripts/External Resources/IconType จากไฟล์ HTML ที่บันทึกไว้
# - พยายามหาไฟล์ด้วยการ match โดเมน (row_*_{domain}*.html/.txt/.docx etc.)
# - ถ้าไม่พบ fallback จับคู่ตาม index ของ URL ↔ ไฟล์ที่ sort แล้ว
# บันทึกผลเป็น CSV คอลัมน์เดียว data_string (หนึ่งแถวต่อหนึ่ง URL)
# -----------------------------------------------------------------------------

import os
import re
import csv
import time
import glob
from pathlib import Path
from datetime import datetime, date
from urllib.parse import urlparse, urljoin

from bs4 import BeautifulSoup
import pandas as pd

# ติดตั้งก่อนใช้งาน:
# pip3.11 install python-whois
try:
    import whois
except Exception as e:
    raise SystemExit("❌ ต้องติดตั้ง python-whois ก่อน:  pip3.11 install python-whois") from e

# ไม่บังคับ (อ่าน .docx)
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    print("⚠️  mammoth not installed - .docx files will be skipped. Install with: pip3.11 install mammoth")


# ===================== Utilities =====================

def load_urls_from_file(filename: str) -> list[str]:
    """อ่าน URL จากไฟล์ paste-* (หนึ่ง URL ต่อบรรทัด)"""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip()]
        print(f"✅ Loaded {len(urls)} URLs from {filename}")
        return urls
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return []


def clean_value(v) -> str:
    if v is None:
        return ""
    s = str(v).replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", s)


def extract_domain_from_url(url: str) -> str:
    if not url or url == "unknown_url":
        return "unknown_domain"
    try:
        return urlparse(url).netloc
    except Exception:
        return "unknown_domain"


def try_parse_date(s: str):
    s = (s or "").strip()
    fmts = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d", "%d-%b-%Y", "%m/%d/%Y", "%d/%m/%Y",
        "%Y/%m/%d", "%Y.%m.%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


# ===================== Analyzer Class =====================

class PasteURLHTMLAnalyzer:
    def __init__(self,
                 reference_date=date(2025, 1, 1),
                 blacklist_urls_file="paste-2.txt",
                 whitelist_urls_file="paste-3.txt",
                 blacklist_html_dir="training/blacktrain",
                 whitelist_html_dir="training/whitetrain"):
        self.reference_date = reference_date
        self.blacklist_urls_file = blacklist_urls_file
        self.whitelist_urls_file = whitelist_urls_file
        self.blacklist_html_dir = blacklist_html_dir
        self.whitelist_html_dir = whitelist_html_dir
        self._whois_cache: dict[str, float | None] = {}

    # ---------- WHOIS ----------
    def get_domain_age_years_label(self, domain: str) -> str:
        """
        คืนเป็น "X.YZ years" หรือ "suspect"
        กติกา: ไม่เจอ/ผิดพลาด/<=0 => suspect
        """
        try:
            if not domain or domain == "unknown_domain" or "." not in domain:
                return "suspect"

            host = domain.lower()
            if host.startswith("www."):
                host = host[4:]
            host = host.split("/")[0]

            if host in self._whois_cache:
                age = self._whois_cache[host]
                return f"{age:.2f} years" if age and age > 0 else "suspect"

            print(f"🔍 WHOIS lookup: {host}")
            info = whois.whois(host)

            creation_date = None
            if hasattr(info, "creation_date") and info.creation_date:
                creation_date = info.creation_date
                if isinstance(creation_date, list) and creation_date:
                    creation_date = creation_date[0]
                if isinstance(creation_date, str):
                    parsed = try_parse_date(creation_date)
                    creation_date = parsed if parsed else None
                if isinstance(creation_date, datetime):
                    creation_date = creation_date.date()

            if not creation_date:
                print("❌ Creation date not found in WHOIS")
                self._whois_cache[host] = None
                return "suspect"

            age_days = (self.reference_date - creation_date).days
            age_years = round(age_days / 365.25, 2)
            print(f"✅ Domain created: {creation_date} → age {age_years} years")
            self._whois_cache[host] = age_years
            return f"{age_years:.2f} years" if age_years > 0 else "suspect"

        except Exception as e:
            print(f"❌ WHOIS failed for {domain}: {e}")
            return "suspect"

    # ---------- File loading ----------
    @staticmethod
    def safe_read_text(filepath: str) -> str | None:
        try:
            ext = Path(filepath).suffix.lower()
            if ext == ".docx":
                return None  # handle by docx method
            encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "windows-1252", "cp1251"]
            for enc in encodings:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            return None
        except Exception:
            return None

    @staticmethod
    def read_html_from_file(filepath: str) -> str | None:
        p = Path(filepath)
        ext = p.suffix.lower()
        print(f"📄 Reading: {p.name} ({ext})")
        if ext in [".html", ".htm", ".txt"]:
            txt = PasteURLHTMLAnalyzer.safe_read_text(str(p))
            if txt is None:
                print("❌ Failed to read text file")
            else:
                print(f"✅ Read {len(txt)} chars")
            return txt
        if ext == ".docx":
            if not MAMMOTH_AVAILABLE:
                print("⚠️ mammoth not installed; skip .docx")
                return None
            try:
                with open(p, "rb") as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    html = result.value
                    if result.messages:
                        print(f"⚠️ docx messages: {result.messages}")
                    print(f"✅ Read {len(html)} chars (docx→html)")
                    return html
            except Exception as e:
                print(f"❌ docx read error: {e}")
                return None
        print("❌ Unsupported extension; skip")
        return None

    # ---------- Match URL to saved file ----------
    @staticmethod
    def _candidate_name_patterns(full_domain: str, base_domain: str | None):
        pats = set()
        for d in [full_domain, base_domain]:
            if not d:
                continue
            d = d.lower()
            if d.startswith("www."):
                d = d[4:]
            pats.update({
                d, d.replace(".", "_"),
                d + "_", d.replace(".", "_") + "_"
            })
        return list(pats)

    def find_saved_file_for_url(self, url: str, folder: str) -> str | None:
        """พยายามหาไฟล์จากโดเมนก่อน ถ้าไม่เจอ return None"""
        full_domain = extract_domain_from_url(url)
        base_domain = None
        try:
            parts = full_domain.split(".")
            if len(parts) >= 2:
                base_domain = ".".join(parts[-2:])
        except Exception:
            base_domain = None

        exts = (".html", ".htm", ".txt") + ((".docx",) if MAMMOTH_AVAILABLE else tuple())
        patterns = self._candidate_name_patterns(full_domain, base_domain)
        candidates = []

        for pat in patterns:
            for ext in exts:
                candidates += glob.glob(os.path.join(folder, f"row_*_{pat}*{ext}"))
                candidates += glob.glob(os.path.join(folder, f"*{pat}*{ext}"))

        # rank: html/htm first, then shorter basename
        def rank(p):
            e = Path(p).suffix.lower()
            ext_rank = 0 if e in (".html", ".htm") else 1
            return (ext_rank, len(Path(p).name))

        candidates = sorted(set(candidates), key=rank)
        return candidates[0] if candidates else None

    # ---------- Compose data_string from soup + url ----------
    @staticmethod
    def extract_title(soup: BeautifulSoup) -> str:
        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    @staticmethod
    def analyze_icon_type(soup: BeautifulSoup, base_url: str) -> str:
        sels = [
            'link[rel*="icon"]',
            'link[rel="shortcut icon"]',
            'link[rel="apple-touch-icon"]',
            'link[rel="apple-touch-icon-precomposed"]',
            'link[rel="favicon"]',
            'link[rel="mask-icon"]',
        ]
        try:
            base_domain = urlparse(base_url).netloc if base_url else ""
        except Exception:
            base_domain = ""
        internal = external = 0
        for sel in sels:
            for link in soup.select(sel):
                href = link.get("href")
                if not href:
                    continue
                try:
                    absu = urljoin(base_url if base_url else "http://local/", href)
                    dom = urlparse(absu).netloc
                    if dom and dom != base_domain:
                        external += 1
                    else:
                        internal += 1
                except Exception:
                    continue
        if internal > 0 and external == 0:
            return "internal"
        if external > 0 and internal == 0:
            return "external"
        if internal > 0 and external > 0:
            return "mixed"
        return "none"

    @staticmethod
    def count_scripts(soup: BeautifulSoup, base_url: str) -> dict:
        all_scripts = soup.find_all("script")
        try:
            base_domain = urlparse(base_url).netloc if base_url else ""
        except Exception:
            base_domain = ""
        src_scripts = inline_scripts = external_scripts = 0
        for sc in all_scripts:
            src = sc.get("src")
            if src:
                src_scripts += 1
                try:
                    if src.startswith("http"):
                        dom = urlparse(src).netloc
                        if dom != base_domain:
                            external_scripts += 1
                    elif src.startswith("//"):
                        dom = urlparse("http:" + src).netloc
                        if dom != base_domain:
                            external_scripts += 1
                except Exception:
                    pass
            else:
                inline_scripts += 1
        return {
            "total": len(all_scripts),
            "src": src_scripts,
            "inline": inline_scripts,
            "external": external_scripts
        }

    @staticmethod
    def count_external_resources(soup: BeautifulSoup, base_url: str) -> int:
        try:
            base_domain = urlparse(base_url).netloc if base_url else ""
        except Exception:
            base_domain = ""
        tags = [
            ("script", "src"), ("link", "href"), ("img", "src"),
            ("iframe", "src"), ("embed", "src"), ("object", "data"),
            ("source", "src"), ("track", "src"), ("video", "src"),
            ("audio", "src"), ("input", "src"), ("frame", "src"),
            ("applet", "archive"), ("area", "href"), ("base", "href"),
        ]
        total_external = 0
        for tag, attr in tags:
            for el in soup.find_all(tag):
                src = el.get(attr)
                if not src:
                    continue
                try:
                    is_ext = False
                    if src.startswith("http"):
                        dom = urlparse(src).netloc
                        is_ext = dom != base_domain
                    elif src.startswith("//"):
                        dom = urlparse("http:" + src).netloc
                        is_ext = dom != base_domain
                    if is_ext:
                        total_external += 1
                except Exception:
                    continue
        return total_external

    @staticmethod
    def count_popups(soup: BeautifulSoup) -> int:
        pats = [
            r"window\.open\s*\(", r"alert\s*\(", r"confirm\s*\(", r"prompt\s*\(",
            r"\.modal\s*\(", r"showModal", r"openPopup", r"\.popup\s*\(", r"\.dialog\s*\("
        ]
        n = 0
        for sc in soup.find_all("script"):
            if not sc.string:
                continue
            txt = sc.string
            for pat in pats:
                n += len(re.findall(pat, txt, flags=re.I))
        return n

    def build_data_string(self, url: str, html_text: str) -> str:
        soup = BeautifulSoup(html_text, "html.parser")

        lines = []
        lines.append(f"URL: {url}")
        domain = extract_domain_from_url(url)
        lines.append(f"Domain: {domain}")

        # Domain age (WHOIS); suspect when fail/<=0
        domain_age = self.get_domain_age_years_label(domain)
        lines.append(f"Domain_age: {domain_age}")

        # Title
        lines.append(f"Title: {self.extract_title(soup)}")

        # Meta tags
        meta_tags = soup.find_all("meta")
        print(f"📋 Processing {len(meta_tags)} meta tags")
        for m in meta_tags:
            if m.get("charset"):
                lines.append(f"Meta_charset: {clean_value(m.get('charset'))}")
                continue
            key = None
            if m.get("name"):
                key = "Meta_" + clean_value(m.get("name"))
            elif m.get("property"):
                key = "Meta_" + clean_value(m.get("property"))
            elif m.get("http-equiv"):
                key = "Meta_" + clean_value(m.get("http-equiv"))
            elif m.get("itemprop"):
                key = "Meta_" + clean_value(m.get("itemprop"))
            if key:
                lines.append(f"{key}: {clean_value(m.get('content', ''))}")
            else:
                # fallback: dump attributes
                attrs_list = []
                for k, v in m.attrs.items():
                    ck = clean_value(k)
                    cv = clean_value(v)
                    attrs_list.append(f"{ck}={cv}" if cv else ck)
                if attrs_list:
                    lines.append(f"Meta_{','.join(attrs_list)}:")

        # Icon type
        icon_type = self.analyze_icon_type(soup, url)
        lines.append(f"Icon_type: {icon_type}")

        # Counts
        sc = self.count_scripts(soup, url)
        ext_total = self.count_external_resources(soup, url)
        lines.append("Number_of_redirect: 0")  # local analysis; no redirects
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {sc['total']}")
        lines.append(f"Number_of_src_script: {sc['src']}")
        lines.append(f"Number_of_inline_script: {sc['inline']}")
        lines.append(f"Number_of_external_script: {sc['external']}")
        lines.append(f"Number_of_external_resources: {ext_total}")

        s = "\n".join(lines)
        print("📊 Final:")
        print(f"   • Domain: {domain}")
        print(f"   • Domain age: {domain_age}")
        print(f"   • Meta: {len(meta_tags)} | Scripts: {sc['total']} | Ext res: {ext_total}")
        print(f"   • String length: {len(s)} chars")
        return s

    # ---------- Batch with paste URL list ----------
    def process_with_paste(self, urls_file: str, html_dir: str, list_type: str, out_csv_path: str):
        urls = load_urls_from_file(urls_file)
        files = sorted(glob.glob(os.path.join(html_dir, "*")))
        # filter supported
        supported_exts = {".html", ".htm", ".txt"}
        if MAMMOTH_AVAILABLE:
            supported_exts.add(".docx")
        files = [f for f in files if Path(f).suffix.lower() in supported_exts]

        results = []
        print(f"\n🚀 Processing {list_type.upper()} — URLs: {len(urls)} | Files: {len(files)}")
        for i, url in enumerate(urls, 1):
            print("\n" + "=" * 80)
            print(f"[{i}/{len(urls)}] URL: {url}")

            # 1) try domain-based lookup
            matched_fp = self.find_saved_file_for_url(url, html_dir)
            if matched_fp and os.path.exists(matched_fp):
                print(f"📎 Matched file by domain: {Path(matched_fp).name}")
            else:
                # 2) fallback by index if possible
                idx = i - 1
                matched_fp = files[idx] if idx < len(files) else None
                if matched_fp:
                    print(f"🔁 Fallback by index → {Path(matched_fp).name}")
                else:
                    print("⚠️  No file available to pair; skipping HTML analysis (URL only)")
                    # ยังเขียนแถวให้ก็ได้ โดยใช้แค่ URL/Domain/Domain_age และค่าอื่นว่าง
                    domain = extract_domain_from_url(url)
                    domain_age = self.get_domain_age_years_label(domain)
                    minimal = "\n".join([
                        f"URL: {url}",
                        f"Domain: {domain}",
                        f"Domain_age: {domain_age}",
                        "Title: ",
                        "Icon_type: none",
                        "Number_of_redirect: 0",
                        "Number_of_popup: 0",
                        "Number_of_script: 0",
                        "Number_of_src_script: 0",
                        "Number_of_inline_script: 0",
                        "Number_of_external_script: 0",
                        "Number_of_external_resources: 0",
                    ])
                    results.append(minimal)
                    continue

            html_text = self.read_html_from_file(matched_fp)
            if not html_text:
                print("❌ Could not read matched file; writing minimal row")
                domain = extract_domain_from_url(url)
                domain_age = self.get_domain_age_years_label(domain)
                minimal = "\n".join([
                    f"URL: {url}",
                    f"Domain: {domain}",
                    f"Domain_age: {domain_age}",
                    "Title: ",
                    "Icon_type: none",
                    "Number_of_redirect: 0",
                    "Number_of_popup: 0",
                    "Number_of_script: 0",
                    "Number_of_src_script: 0",
                    "Number_of_inline_script: 0",
                    "Number_of_external_script: 0",
                    "Number_of_external_resources: 0",
                ])
                results.append(minimal)
                continue

            data_string = self.build_data_string(url, html_text)
            results.append(data_string)
            time.sleep(0.05)

        # save one CSV with single column data_string
        out_dir = Path(out_csv_path).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame({"data_string": results})
        df.to_csv(out_csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        print(f"\n💾 Saved {len(df)} rows → {out_csv_path}")

    # ---------- Entry point ----------
    def run(self):
        print("🎯 Start processing with pasted URLs…")
        # BLACK
        self.process_with_paste(
            urls_file=self.blacklist_urls_file,
            html_dir=self.blacklist_html_dir,
            list_type="black",
            out_csv_path="csv_train/black/black_data_strings.csv"
        )
        # WHITE
        self.process_with_paste(
            urls_file=self.whitelist_urls_file,
            html_dir=self.whitelist_html_dir,
            list_type="white",
            out_csv_path="csv_train/white/white_data_strings.csv"
        )
        print("\n🎉 Done. Check csv_train/black/ and csv_train/white/.")


# ===================== Main =====================

if __name__ == "__main__":
    analyzer = PasteURLHTMLAnalyzer(
        reference_date=date(2025, 1, 1),
        blacklist_urls_file="paste-2.txt",
        whitelist_urls_file="paste-3.txt",
        blacklist_html_dir="training/blacktrain",
        whitelist_html_dir="training/whitetrain",
    )
    analyzer.run()
