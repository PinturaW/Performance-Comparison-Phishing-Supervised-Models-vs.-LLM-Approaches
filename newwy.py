# paser.py — save-per-file with leftover sweep
# -----------------------------------------------------------------------------
# - Read URLs from paste-2.txt (black) / paste-3.txt (white)
# - Produce one CSV per URL/HTML: csv_train_new/{black|white}/{stem}_analyzed.csv
# - WHOIS domain age; missing/0/error => 'suspect'
# - Extract Title / Meta / Icon type / counts
# - NEW: also process leftover files not matched by any URL (optional)
# - NEW: option to run only white or only black
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

# ======= CONFIG =======
ONLY_WHITE = False       # True = ประมวลผลเฉพาะ white
ONLY_BLACK = False       # True = ประมวลผลเฉพาะ black
INCLUDE_LEFTOVER_FILES = True  # กวาดไฟล์ที่ไม่ถูกจับคู่ด้วยหรือไม่
SLEEP_BETWEEN = 0.03     # เวลาหน่วงเล็กน้อยระหว่างงาน

# WHOIS (required)
try:
    import whois
except Exception as e:
    raise SystemExit("❌ Please install:  pip3.11 install python-whois") from e

# Optional: read .docx saved pages
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    print("⚠️  'mammoth' not installed — .docx will be skipped (pip3.11 install mammoth)")


# ------------------------- helpers -------------------------

def load_urls_from_file(filename):
    """Load URLs (one per line) from paste file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            urls = [u.strip() for u in f if u.strip()]
        print(f"✅ Loaded {len(urls)} URLs from {filename}")
        return urls
    except FileNotFoundError:
        print(f"❌ File not found: {filename}")
        return []


def clean_value(v):
    if v is None:
        return ""
    s = str(v).replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", s)


def extract_domain_from_url(url):
    if not url or url == "unknown_url":
        return "unknown_domain"
    try:
        return urlparse(url).netloc
    except Exception:
        return "unknown_domain"


def try_parse_date(s):
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


def make_safe_stem_from_url(url, idx):
    dom = extract_domain_from_url(url).lower()
    dom = dom.replace("www.", "")
    dom = re.sub(r"[^a-z0-9.\-_]+", "_", dom)
    return f"url_{idx:04d}_{dom or 'unknown'}"


# ------------------------- analyzer -------------------------

class PasteURLHTMLAnalyzer:
    def __init__(self,
                 reference_date=date(2025, 1, 1),
                 blacklist_urls_file="paste-2.txt",
                 whitelist_urls_file="paste-3.txt",
                 blacklist_html_dir="train_new/blackTraining",
                 whitelist_html_dir="train_new/whiteTraining"):
        self.reference_date = reference_date
        self.blacklist_urls_file = blacklist_urls_file
        self.whitelist_urls_file = whitelist_urls_file
        self.blacklist_html_dir = blacklist_html_dir
        self.whitelist_html_dir = whitelist_html_dir
        self._whois_cache = {}

    # -------- WHOIS: return "X.YY years" or "suspect" --------
    def get_domain_age_years_label(self, domain):
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

            print(f"🔍 WHOIS: {host}")
            info = whois.whois(host)

            creation_date = getattr(info, "creation_date", None)
            if isinstance(creation_date, list) and creation_date:
                creation_date = creation_date[0]
            if isinstance(creation_date, str):
                parsed = try_parse_date(creation_date)
                creation_date = parsed if parsed else None
            if isinstance(creation_date, datetime):
                creation_date = creation_date.date()

            if not creation_date:
                print("❌ WHOIS creation_date missing")
                self._whois_cache[host] = None
                return "suspect"

            age_days = (self.reference_date - creation_date).days
            age_years = round(age_days / 365.25, 2)
            self._whois_cache[host] = age_years
            return f"{age_years:.2f} years" if age_years > 0 else "suspect"

        except Exception as e:
            print(f"❌ WHOIS failed for {domain}: {e}")
            return "suspect"

    # -------- extract URL from HTML content --------
    @staticmethod
    def extract_url_from_html(html_text: str) -> str:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            # canonical
            c = soup.find("link", rel=lambda v: v and "canonical" in v.lower())
            if c and c.get("href"):
                return c.get("href").strip()
            # og:url
            og = soup.find("meta", property=lambda v: v and v.lower() == "og:url")
            if og and og.get("content"):
                return og.get("content").strip()
            # base
            b = soup.find("base")
            if b and b.get("href"):
                return b.get("href").strip()
        except Exception:
            pass
        return "unknown_url"

    # -------- read saved HTML --------
    @staticmethod
    def safe_read_text(filepath):
        try:
            ext = Path(filepath).suffix.lower()
            if ext == ".docx":
                return None
            encs = ["utf-8", "utf-8-sig", "iso-8859-1", "windows-1252", "cp1251"]
            for enc in encs:
                try:
                    with open(filepath, "r", encoding=enc) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            return None
        except Exception:
            return None

    @staticmethod
    def read_html_from_file(filepath):
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
                    print(f"✅ Read {len(html)} chars (docx→html)")
                    return html
            except Exception as e:
                print(f"❌ .docx read error: {e}")
                return None
        print("❌ Unsupported extension; skip")
        return None

    # -------- find saved file for URL --------
    @staticmethod
    def _candidate_name_patterns(full_domain, base_domain):
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

    def find_saved_file_for_url(self, url, folder):
        full_domain = extract_domain_from_url(url)
        parts = full_domain.split(".")
        base_domain = ".".join(parts[-2:]) if len(parts) >= 2 else full_domain

        exts = (".html", ".htm", ".txt") + ((".docx",) if MAMMOTH_AVAILABLE else tuple())
        patterns = self._candidate_name_patterns(full_domain, base_domain)
        candidates = []

        for pat in patterns:
            for ext in exts:
                candidates += glob.glob(os.path.join(folder, f"row_*_{pat}*{ext}"))
                candidates += glob.glob(os.path.join(folder, f"*{pat}*{ext}"))

        # Prefer html/htm and shorter names
        def rank(p):
            e = Path(p).suffix.lower()
            ext_rank = 0 if e in (".html", ".htm") else 1
            return (ext_rank, len(Path(p).name))

        candidates = sorted(set(candidates), key=rank)
        return candidates[0] if candidates else None

    # -------- features --------
    @staticmethod
    def extract_title(soup):
        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    @staticmethod
    def analyze_icon_type(soup, base_url):
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
    def count_scripts(soup, base_url):
        alls = soup.find_all("script")
        try:
            base_domain = urlparse(base_url).netloc if base_url else ""
        except Exception:
            base_domain = ""
        src_scripts = inline_scripts = external_scripts = 0
        for sc in alls:
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
        return {"total": len(alls), "src": src_scripts, "inline": inline_scripts, "external": external_scripts}

    @staticmethod
    def count_external_resources(soup, base_url):
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
    def count_redirects(soup, base_url):
        redirect_count = 0

        # Detect meta refresh redirects
        meta_refreshes = soup.find_all("meta", {"http-equiv": "refresh"})
        redirect_count += len(meta_refreshes)

        # Detect javascript/url-based redirects
        scripts = soup.find_all("script")
        js_redirect_patterns = [
            r"window\.location\s*=", r"window\.location\.href\s*=",
            r"window\.location\.replace\s*\(", r"document\.location\s*=",
            r"location\.href\s*=", r"location\.replace\s*\(",
            r"window\.open\s*\("
        ]
        for script in scripts:
            if script.string:
                for pattern in js_redirect_patterns:
                    redirect_count += len(re.findall(pattern, script.string, flags=re.I))

        # Detect form action redirects to external domain
        forms = soup.find_all("form")
        for form in forms:
            action = form.get("action")
            if action and action.startswith("http"):
                try:
                    form_domain = urlparse(action).netloc
                    base_domain = urlparse(base_url).netloc if base_url else ""
                    if form_domain != base_domain:
                        redirect_count += 1
                except:
                    pass

        return redirect_count

    @staticmethod
    def count_popups(soup):
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

    # -------- build data_string --------
    def build_data_string(self, url, html_text):
        soup = BeautifulSoup(html_text, "html.parser")
        lines = []
        lines.append(f"URL: {url}")
        domain = extract_domain_from_url(url)
        lines.append(f"Domain: {domain}")
        lines.append(f"Domain_age: {self.get_domain_age_years_label(domain)}")
        lines.append(f"Title: {self.extract_title(soup)}")

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

        icon_type = self.analyze_icon_type(soup, url)
        lines.append(f"Icon_type: {icon_type}")

        sc = self.count_scripts(soup, url)
        ext_total = self.count_external_resources(soup, url)
        redirect_count = self.count_redirects(soup, url)
        lines.append("Number_of_redirect: {}".format(redirect_count))
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {sc['total']}")
        lines.append(f"Number_of_src_script: {sc['src']}")
        lines.append(f"Number_of_inline_script: {sc['inline']}")
        lines.append(f"Number_of_external_script: {sc['external']}")
        lines.append(f"Number_of_external_resources: {ext_total}")

        return "\n".join(lines)

    # -------- save ONE row per file --------
    def save_to_csv(self, data_string, filename, list_type):
        """Save 1-row CSV with a single 'data_string' column."""
        folder_path = f'csv_train_new/{list_type}'
        os.makedirs(folder_path, exist_ok=True)

        safe_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', filename).strip('_')
        if not safe_name.endswith('.csv'):
            safe_name += '.csv'

        filepath = os.path.join(folder_path, safe_name)
        df = pd.DataFrame({"data_string": [data_string]})
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        print(f"💾 Saved to: {filepath}")

    # -------- main processing for one list --------
    def process_with_paste(self, urls_file, html_dir, list_type):
        urls = load_urls_from_file(urls_file)
        files = sorted(glob.glob(os.path.join(html_dir, "*")))
        supported_exts = {".html", ".htm", ".txt"}
        if MAMMOTH_AVAILABLE:
            supported_exts.add(".docx")
        files = [f for f in files if Path(f).suffix.lower() in supported_exts]

        print(f"\n🚀 {list_type.upper()} → URLs: {len(urls)} | Files: {len(files)}")
        seen_files = set()
        save_count = 0

        # pass 1: by URLs (authoritative)
        for i, url in enumerate(urls, 1):
            print("\n" + "=" * 80)
            print(f"[{i}/{len(urls)}] URL: {url}")

            matched_fp = self.find_saved_file_for_url(url, html_dir)
            if matched_fp and os.path.exists(matched_fp):
                stem = Path(matched_fp).stem
                seen_files.add(os.path.abspath(matched_fp))
                print(f"📎 Matched by domain: {Path(matched_fp).name}")
            else:
                idx = i - 1
                if idx < len(files):
                    matched_fp = files[idx]
                    stem = Path(matched_fp).stem
                    seen_files.add(os.path.abspath(matched_fp))
                    print(f"🔁 Fallback by index → {Path(matched_fp).name}")
                else:
                    # no file at all → minimal row
                    stem = make_safe_stem_from_url(url, i)
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
                    self.save_to_csv(minimal, stem + "_analyzed", list_type)
                    save_count += 1
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
                self.save_to_csv(minimal, stem + "_analyzed", list_type)
                save_count += 1
                continue

            data_string = self.build_data_string(url, html_text)
            self.save_to_csv(data_string, stem + "_analyzed", list_type)
            save_count += 1
            time.sleep(SLEEP_BETWEEN)

        # pass 2: leftover files (no URL matched)
        if INCLUDE_LEFTOVER_FILES:
            leftovers = [f for f in files if os.path.abspath(f) not in seen_files]
            if leftovers:
                print(f"\n🧹 Processing leftover files not in paste ({len(leftovers)} files)…")
            for j, fp in enumerate(leftovers, 1):
                stem = Path(fp).stem
                html_text = self.read_html_from_file(fp)
                if not html_text:
                    minimal = "\n".join([
                        "URL: unknown_url",
                        "Domain: unknown_domain",
                        "Domain_age: suspect",
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
                    self.save_to_csv(minimal, stem + "_analyzed", list_type)
                    save_count += 1
                    continue

                url_guess = self.extract_url_from_html(html_text)
                data_string = self.build_data_string(url_guess, html_text)
                self.save_to_csv(data_string, stem + "_analyzed", list_type)
                save_count += 1
                time.sleep(SLEEP_BETWEEN)

        print(f"\n✅ {list_type.upper()} saved rows: {save_count}")

    # -------- entrypoint --------
    def run(self):
        print("🎯 Start")
        if not ONLY_WHITE:
            self.process_with_paste(self.blacklist_urls_file, self.blacklist_html_dir, "black")
        if not ONLY_BLACK:
            self.process_with_paste(self.whitelist_urls_file, self.whitelist_html_dir, "white")
        print("\n🎉 Done. Check csv_train_new/{black,white}/*.csv")


# ------------------------- main -------------------------

if __name__ == "__main__":
    analyzer = PasteURLHTMLAnalyzer(
        reference_date=date(2025, 1, 1),
        blacklist_urls_file="black_url.txt",
        whitelist_urls_file="white_url.txt",
        blacklist_html_dir="train_new/blackTraining",
        whitelist_html_dir="train_new/whiteTraining",
    )
    analyzer.run()