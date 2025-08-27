# readnew.py
# -----------------------------------------------------------------------------
# วิเคราะห์ไฟล์ HTML ที่บันทึกไว้ (blacktrain / whitetrain)
# - ดึง URL (ถ้ามี) จาก canonical / og:url / base / title / comments
# - อ่าน Domain Age จาก WHOIS (reference_date = 2025-01-01)
#   * ถ้าหาไม่เจอ / error / ได้อายุ <= 0 → "Domain_age: suspect"
# - ดึง Title, Meta ทั้งหมด, Icon type, จำนวน scripts / external resources / popups
# - บันทึกผลเป็น CSV คอลัมน์เดียว: data_string ไปที่ csv_train/{black,white}/
# -----------------------------------------------------------------------------

import os
import re
import csv
import time
import glob
import random
from pathlib import Path
from datetime import datetime, date
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd

# whois (python-whois) ต้องติดตั้งก่อน: pip3.11 install python-whois
try:
    import whois
except Exception as e:
    raise SystemExit("❌ ต้องติดตั้ง python-whois ก่อน:  pip3.11 install python-whois") from e

# .docx (ไม่บังคับ)
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    print("⚠️  mammoth not installed - .docx files will be skipped. Install with: pip3.11 install mammoth")

class HTMLFileAnalyzer:
    def __init__(self):
        # อ้างอิงอายุโดเมนถึง 1 ม.ค. 2025 ตามที่ตั้งไว้
        self.reference_date = date(2025, 1, 1)
        # cache ลดการยิง WHOIS ซ้ำ
        self._whois_cache = {}

    # ----------------- I/O: read file -----------------
    def read_html_from_file(self, filepath: str):
        try:
            file_path = Path(filepath)
            ext = file_path.suffix.lower()
            print(f"📄 Reading file: {filepath} (type: {ext})")

            if ext == ".docx":
                if not MAMMOTH_AVAILABLE:
                    print("❌ .docx not supported (install mammoth)")
                    return None
                with open(filepath, "rb") as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    html_content = result.value
                    if result.messages:
                        print(f"⚠️  Docx conversion messages: {result.messages}")
                print(f"📊 HTML content length: {len(html_content)} characters")
                return html_content

            if ext in [".html", ".htm", ".txt"]:
                encodings = ["utf-8", "utf-8-sig", "iso-8859-1", "windows-1252", "cp1251"]
                for enc in encodings:
                    try:
                        with open(filepath, "r", encoding=enc) as f:
                            html_content = f.read()
                        print(f"✅ Successfully read with encoding: {enc}")
                        print(f"📊 HTML content length: {len(html_content)} characters")
                        return html_content
                    except UnicodeDecodeError:
                        continue
                print("❌ Failed to read file with any encoding")
                return None

            print(f"❌ Unsupported file type: {ext}")
            return None
        except Exception as e:
            print(f"❌ Error reading file {filepath}: {e}")
            return None

    # ----------------- URL extraction -----------------
    def extract_url_from_content(self, html_content: str) -> str:
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            canonical = soup.find("link", rel="canonical")
            if canonical and canonical.get("href"):
                return canonical.get("href")

            og_url = soup.find("meta", property="og:url")
            if og_url and og_url.get("content"):
                return og_url.get("content")

            base_tag = soup.find("base")
            if base_tag and base_tag.get("href"):
                return base_tag.get("href")

            title = soup.find("title")
            if title and title.get_text():
                urls = self._find_urls_in_text(title.get_text())
                if urls:
                    return self._normalize_url(urls[0])

            # scan all text nodes for a url-ish string (เบา ๆ)
            texts = soup.find_all(string=True)
            for t in texts:
                if "http" in t or "www." in t:
                    urls = self._find_urls_in_text(str(t))
                    if urls:
                        return self._normalize_url(urls[0])

            return "unknown_url"
        except Exception as e:
            print(f"⚠️  Could not extract URL from content: {e}")
            return "unknown_url"

    @staticmethod
    def _find_urls_in_text(text: str):
        pat = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+\.[^\s<>"\']+'
        return re.findall(pat, text)

    @staticmethod
    def _normalize_url(u: str) -> str:
        return u if u.startswith("http") else f"https://{u}"

    # ----------------- Domain age (WHOIS) -----------------
    def get_domain_age(self, domain: str) -> str:
        """
        return "X.YZ years" หรือ "suspect" ตามนโยบาย:
        - หาไม่เจอ / error / อายุ <= 0 → "suspect"
        """
        try:
            if domain == "unknown_domain" or not domain or "." not in domain:
                return "suspect"

            host = domain.lower()
            if host.startswith("www."):
                host = host[4:]
            host = host.split("/")[0]  # กันเผื่อมี path ติดมา

            if host in self._whois_cache:
                age_years = self._whois_cache[host]
                return f"{age_years:.2f} years" if age_years and age_years > 0 else "suspect"

            print(f"🔍 WHOIS lookup: {host}")
            info = whois.whois(host)

            creation_date = None
            if hasattr(info, "creation_date") and info.creation_date:
                creation_date = info.creation_date
                if isinstance(creation_date, list) and creation_date:
                    creation_date = creation_date[0]

                # บาง registrar คืน string
                if isinstance(creation_date, str):
                    parsed = self._try_parse_date_str(creation_date)
                    if parsed:
                        creation_date = parsed
                    else:
                        creation_date = None

                # ให้เป็น date
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
            print(f"❌ WHOIS lookup failed for {domain}: {e}")
            return "suspect"

    @staticmethod
    def _try_parse_date_str(s: str):
        s = s.strip()
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

    # ----------------- Core analysis -----------------
    def analyze_html_file(self, filepath: str, list_type: str = "white"):
        try:
            html_content = self.read_html_from_file(filepath)
            if not html_content:
                return None

            soup = BeautifulSoup(html_content, "html.parser")
            extracted_url = self.extract_url_from_content(html_content)
            print(f"🔗 Extracted URL: {extracted_url}")

            result_string = self.create_line_by_line_string(soup, extracted_url)
            result = {"data_string": result_string}

            out_name = f"{Path(filepath).stem}_analyzed"
            self.save_to_csv(result, out_name, list_type)

            print(f"✅ Analysis completed for: {filepath}")
            return result
        except Exception as e:
            print(f"❌ Error analyzing {filepath}: {e}")
            return None

    def batch_analyze_files(self, folder_path: str, list_type: str = "white", file_pattern: str = "*"):
        try:
            folder = Path(folder_path)
            if not folder.exists():
                print(f"❌ Folder not found: {folder}")
                return None

            exts = ["*.html", "*.htm", "*.txt"]
            if MAMMOTH_AVAILABLE:
                exts.append("*.docx")

            files = []
            for ext in exts:
                files.extend(glob.glob(str(folder / ext)))

            if file_pattern != "*":
                files = [f for f in files if file_pattern in Path(f).name]

            print(f"\n🚀 Starting batch analysis for {len(files)} files ({list_type}list)")
            print(f"📁 Source folder: {folder}")
            print(f"📁 Output folder: csv_train/{list_type}/")

            ok = fail = 0
            for i, fp in enumerate(files, 1):
                print("\n" + "=" * 80)
                print(f"📊 Progress: {i}/{len(files)} ({(i/len(files)*100):.1f}%)")
                print(f"📄 Processing: {Path(fp).name}")
                try:
                    res = self.analyze_html_file(fp, list_type)
                    if res: ok += 1
                    else: fail += 1
                except Exception as e:
                    fail += 1
                    print(f"❌ Error with {Path(fp).name}: {e}")
                if i < len(files):
                    time.sleep(0.05)

            print("\n" + "=" * 80)
            print("🎉 Batch analysis completed!")
            print(f"✅ Successful: {ok}")
            print(f"❌ Failed: {fail}")
            rate = (ok / (ok + fail) * 100) if (ok + fail) else 0.0
            print(f"📊 Success rate: {rate:.1f}%")

            return {"success_count": ok, "failed_count": fail, "total": len(files)}
        except Exception as e:
            print(f"❌ Error in batch analysis: {e}")
            return None

    def process_training_data_from_files(self, blacklist_folder="training/blacktrain", whitelist_folder="training/whitetrain"):
        print("🎯 Starting training data processing from files...")
        results = {}
        if os.path.exists(blacklist_folder):
            print(f"\n🔴 Processing BLACKLIST files from: {blacklist_folder}")
            results["blacklist"] = self.batch_analyze_files(blacklist_folder, "black")
        else:
            print(f"⚠️  Blacklist folder not found: {blacklist_folder}")
            results["blacklist"] = {"success_count": 0, "failed_count": 0, "total": 0}

        if os.path.exists(whitelist_folder):
            print(f"\n🔵 Processing WHITELIST files from: {whitelist_folder}")
            results["whitelist"] = self.batch_analyze_files(whitelist_folder, "white")
        else:
            print(f"⚠️  Whitelist folder not found: {whitelist_folder}")
            results["whitelist"] = {"success_count": 0, "failed_count": 0, "total": 0}

        self.print_training_summary(results)
        return results

    def print_training_summary(self, results: dict):
        bl = results.get("blacklist", {"success_count": 0, "failed_count": 0, "total": 0})
        wl = results.get("whitelist", {"success_count": 0, "failed_count": 0, "total": 0})

        print("\n" + "=" * 80)
        print("🎉 TRAINING DATA PROCESSING COMPLETED!")
        print("=" * 80)
        print("🔴 BLACKLIST Results:")
        print(f"   ✅ Successful: {bl['success_count']}")
        print(f"   ❌ Failed: {bl['failed_count']}")
        print(f"   📊 Total: {bl['total']}")
        if bl["total"]:
            print(f"   📈 Success rate: {bl['success_count']/bl['total']*100:.1f}%")
        else:
            print("   📈 Success rate: N/A")

        print("\n🔵 WHITELIST Results:")
        print(f"   ✅ Successful: {wl['success_count']}")
        print(f"   ❌ Failed: {wl['failed_count']}")
        print(f"   📊 Total: {wl['total']}")
        if wl["total"]:
            print(f"   📈 Success rate: {wl['success_count']/wl['total']*100:.1f}%")
        else:
            print("   📈 Success rate: N/A")

        total_ok = bl["success_count"] + wl["success_count"]
        total_all = bl["total"] + wl["total"]
        print("\n📊 OVERALL STATISTICS:")
        print(f"   🎯 Total files processed: {total_all}")
        print(f"   ✅ Total successful: {total_ok}")
        if total_all:
            print(f"   📈 Overall success rate: {total_ok/total_all*100:.1f}%")
        print("   📁 Files saved in: csv_train/black/ and csv_train/white/")

    # ----------------- Compose data_string -----------------
    def create_line_by_line_string(self, soup: BeautifulSoup, url: str) -> str:
        lines = []

        # URL + Domain
        lines.append(f"URL: {url}")
        domain = self.extract_domain(url)
        lines.append(f"Domain: {domain}")

        # Domain Age: ใช้ suspect เมื่อหาไม่ได้/0/ผิดพลาด
        domain_age = self.get_domain_age(domain)
        lines.append(f"Domain_age: {domain_age}")

        # Title
        lines.append(f"Title: {self.extract_title(soup)}")

        # Meta
        meta_tags = soup.find_all("meta")
        print(f"\n📋 Processing {len(meta_tags)} meta tags:")
        for m in meta_tags:
            if m.get("charset"):
                lines.append(f"Meta_charset: {self.clean_value(m.get('charset'))}")
                continue

            key = None
            if m.get("name"):
                key = "Meta_" + self.clean_value(m.get("name"))
            elif m.get("property"):
                key = "Meta_" + self.clean_value(m.get("property"))
            elif m.get("http-equiv"):
                key = "Meta_" + self.clean_value(m.get("http-equiv"))
            elif m.get("itemprop"):
                key = "Meta_" + self.clean_value(m.get("itemprop"))

            if key:
                lines.append(f"{key}: {self.clean_value(m.get('content', ''))}")
            else:
                # fallback บันทึกแอตทริบิวต์ทั้งหมด
                attrs_list = []
                for k, v in m.attrs.items():
                    ck = self.clean_value(k)
                    cv = self.clean_value(v)
                    attrs_list.append(f"{ck}={cv}" if cv else ck)
                if attrs_list:
                    lines.append(f"Meta_{','.join(attrs_list)}:")

        # Icon type
        lines.append(f"Icon_type: {self.analyze_icon_type(soup, url)}")

        # Counts
        sc = self.count_all_scripts(soup, url)
        res = self.count_all_external_resources(soup, url)
        lines.append("Number_of_redirect: 0")  # local file → ไม่มี redirect
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {sc['total_scripts']}")
        lines.append(f"Number_of_src_script: {sc['src_scripts']}")
        lines.append(f"Number_of_inline_script: {sc['inline_scripts']}")
        lines.append(f"Number_of_external_script: {sc['external_scripts']}")
        lines.append(f"Number_of_external_resources: {res['total_external']}")

        result_string = "\n".join(lines)

        print("📊 Final statistics:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • Domain: {domain}")
        print(f"   • Domain age: {domain_age}")
        print(f"   • Meta tags: {len(meta_tags)}")
        print(f"   • Scripts: {sc['total_scripts']}")
        print(f"   • External resources: {res['total_external']}")
        print(f"   • String length: {len(result_string)} characters")

        return result_string

    # ----------------- small helpers -----------------
    @staticmethod
    def extract_domain(url: str) -> str:
        if url == "unknown_url" or not url:
            return "unknown_domain"
        try:
            return urlparse(url).netloc
        except Exception:
            return "unknown_domain"

    @staticmethod
    def extract_title(soup: BeautifulSoup) -> str:
        t = soup.find("title")
        return t.get_text(strip=True) if t else ""

    @staticmethod
    def clean_value(v) -> str:
        if v is None:
            return ""
        s = str(v).replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ").strip()
        return re.sub(r"\s+", " ", s)

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
            base_domain = urlparse(base_url).netloc if base_url and base_url != "unknown_url" else ""
        except Exception:
            base_domain = ""
        internal = external = 0
        for sel in sels:
            for link in soup.select(sel):
                href = link.get("href")
                if not href:
                    continue
                try:
                    absu = urljoin(base_url if base_url != "unknown_url" else "http://local/", href)
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
    def count_all_scripts(soup: BeautifulSoup, base_url: str) -> dict:
        all_scripts = soup.find_all("script")
        try:
            base_domain = urlparse(base_url).netloc if base_url and base_url != "unknown_url" else ""
        except Exception:
            base_domain = ""

        src_scripts = 0
        inline_scripts = 0
        external_scripts = 0

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
            "total_scripts": len(all_scripts),
            "src_scripts": src_scripts,
            "inline_scripts": inline_scripts,
            "external_scripts": external_scripts,
        }

    @staticmethod
    def count_all_external_resources(soup: BeautifulSoup, base_url: str) -> dict:
        try:
            base_domain = urlparse(base_url).netloc if base_url and base_url != "unknown_url" else ""
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

        return {"total_external": total_external}

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

    # ----------------- Save CSV -----------------
    @staticmethod
    def save_to_csv(data: dict, filename: str, list_type: str):
        folder = f"csv_train/{list_type}"
        os.makedirs(folder, exist_ok=True)
        if not filename.endswith(".csv"):
            filename += ".csv"
        outpath = f"{folder}/{filename}"
        df = pd.DataFrame([data])
        df.to_csv(outpath, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)
        print(f"💾 Saved to: {outpath}")
        preview = data["data_string"][:200] + "..." if len(data["data_string"]) > 200 else data["data_string"]
        print(f"📋 Data preview: {preview}")


# ----------------- Runner -----------------
if __name__ == "__main__":
    analyzer = HTMLFileAnalyzer()

    print("🎯 HTML File Analyzer")
    print("=" * 50)

    # ประมวลผล training จากโฟลเดอร์ (ตามที่คุณใช้)
    results = analyzer.process_training_data_from_files(
        blacklist_folder="training/blacktrain",
        whitelist_folder="training/whitetrain"
    )

    print("\n🎉 Analysis completed! Check csv_train/ folder for results.")
