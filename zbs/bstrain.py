import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
import os
import csv
import time
import random
import whois
from datetime import datetime, date

class HTMLAnalyzerComplete:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.redirects_count = 0
        self.reference_date = date(2025, 1, 1)

    def analyze_html_file(self, filepath, filename=None, list_type='white'):
        """วิเคราะห์ HTML จากไฟล์ (เช่นจาก Google Drive ที่โหลดมาเก็บไว้แล้ว)"""
        try:
            print(f"📄 Reading local HTML file: {filepath}")
            with open(filepath, 'r', encoding='utf-8') as file:
                html_content = file.read()
            soup = BeautifulSoup(html_content, 'lxml')
            pseudo_url = f"file://{filepath}"
            if filename is None:
                filename = os.path.splitext(os.path.basename(filepath))[0]
            result_string = self.create_line_by_line_string(soup, pseudo_url, html_content)
            result = {
                'data_string': result_string
            }
            self.save_to_csv(result, filename, list_type)
            print(f"✅ Local HTML analysis completed for {filename}")
            return result
        except Exception as e:
            print(f"❌ Error analyzing local file {filepath}: {str(e)}")
            return None

    def batch_analyze_html_folder(self, folder_path, list_type='white'):
        """วิเคราะห์ HTML ทีละไฟล์จากโฟลเดอร์"""
        print(f"🚀 Batch analyzing HTML files in folder: {folder_path}")
        html_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.html', '.htm'))]
        print(f"📁 Found {len(html_files)} HTML files")
        success_count = 0
        for i, html_file in enumerate(html_files, 1):
            print(f"\n📄 [{i}/{len(html_files)}] Processing: {html_file}")
            full_path = os.path.join(folder_path, html_file)
            try:
                result = self.analyze_html_file(full_path, filename=html_file.replace('.html','').replace('.htm',''), list_type=list_type)
                if result:
                    success_count += 1
            except Exception as e:
                print(f"❌ Error: {e}")
        print(f"\n🎉 Done! {success_count}/{len(html_files)} files analyzed successfully.")

    def create_line_by_line_string(self, soup, url, raw_html=""):
        lines = []
        lines.append(f"URL: {url}")
        domain = self.extract_domain(url)
        lines.append(f"Domain: {domain}")
        domain_age = self.get_domain_age(domain)
        lines.append(f"Domain_age: {domain_age}")
        title = self.extract_title(soup)
        lines.append(f"Title: {title}")
        lines.append(f"HTML_line_count: {raw_html.count('\n') + 1}")
        meta_tags = soup.find_all('meta')
        print(f"\n📋 Processing all {len(meta_tags)} meta tags:")
        for meta in meta_tags:
            if meta.get('name'):
                lines.append(f"Meta_{meta.get('name')}: {meta.get('content', '')}")
            elif meta.get('property'):
                lines.append(f"Meta_{meta.get('property')}: {meta.get('content', '')}")
            elif meta.get('http-equiv'):
                lines.append(f"Meta_{meta.get('http-equiv')}: {meta.get('content', '')}")
            elif meta.get('charset'):
                lines.append(f"Meta_charset: {meta.get('charset')}")
            elif meta.get('itemprop'):
                lines.append(f"Meta_{meta.get('itemprop')}: {meta.get('content', '')}")
        icon_type = self.analyze_icon_type(soup, url)
        lines.append(f"Icon_type: {icon_type}")
        script_counts = self.count_all_scripts(soup, url)
        resource_counts = self.count_all_external_resources(soup, url)
        lines.append(f"Number_of_redirect: {self.redirects_count}")
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {script_counts['total_scripts']}")
        lines.append(f"Number_of_src_script: {script_counts['src_scripts']}")
        lines.append(f"Number_of_inline_script: {script_counts['inline_scripts']}")
        lines.append(f"Number_of_external_script: {script_counts['external_scripts']}")
        lines.append(f"Number_of_external_resources: {resource_counts['total_external']}")
        result_string = '\n'.join(lines)
        print(f"📊 Final statistics:\n • Total lines: {len(lines)}\n • Domain age: {domain_age}\n • Meta tags: {len(meta_tags)}\n • HTML line count: {raw_html.count('\n') + 1}\n")
        return result_string

    def get_domain_age(self, domain):
        try:
            if domain.startswith('www.'):
                domain = domain[4:]
            domain_info = whois.whois(domain)
            creation_date = domain_info.creation_date
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if isinstance(creation_date, datetime):
                creation_date = creation_date.date()
            if creation_date:
                age_days = (self.reference_date - creation_date).days
                age_years = age_days / 365.25
                return f"{age_years:.2f} years"
            return "suspect"
        except:
            return "suspect"

    def extract_domain(self, url):
        return urlparse(url).netloc

    def extract_title(self, soup):
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""

    def analyze_icon_type(self, soup, base_url):
        selectors = ['link[rel*="icon"]', 'link[rel="shortcut icon"]', 'link[rel="apple-touch-icon"]']
        base_domain = urlparse(base_url).netloc
        internal, external = 0, 0
        for sel in selectors:
            for link in soup.select(sel):
                href = link.get('href')
                if href:
                    abs_url = urljoin(base_url, href)
                    icon_domain = urlparse(abs_url).netloc
                    if icon_domain == base_domain or not icon_domain:
                        internal += 1
                    else:
                        external += 1
        if internal and not external:
            return 'internal'
        elif external and not internal:
            return 'external'
        elif internal and external:
            return 'mixed'
        return 'none'

    def count_all_scripts(self, soup, base_url):
        all_scripts = soup.find_all('script')
        base_domain = urlparse(base_url).netloc
        src_scripts = [s for s in all_scripts if s.get('src')]
        inline_scripts = [s for s in all_scripts if not s.get('src')]
        external_scripts = [s for s in src_scripts if urlparse(s.get('src')).netloc != base_domain]
        return {
            'total_scripts': len(all_scripts),
            'src_scripts': len(src_scripts),
            'inline_scripts': len(inline_scripts),
            'external_scripts': len(external_scripts)
        }

    def count_all_external_resources(self, soup, base_url):
        tags = [('script', 'src'), ('link', 'href'), ('img', 'src'), ('iframe', 'src')]
        base_domain = urlparse(base_url).netloc
        count = 0
        for tag, attr in tags:
            for el in soup.find_all(tag):
                src = el.get(attr)
                if src:
                    src_domain = urlparse(urljoin(base_url, src)).netloc
                    if src_domain and src_domain != base_domain:
                        count += 1
        return {'total_external': count}

    def count_popups(self, soup):
        patterns = [r'window\.open\s*\(', r'alert\s*\(']
        popup_count = 0
        for script in soup.find_all('script'):
            if script.string:
                for pattern in patterns:
                    popup_count += len(re.findall(pattern, script.string, re.IGNORECASE))
        return popup_count

    def save_to_csv(self, data, filename, list_type):
        folder_path = f'csv_training/{list_type}'
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, filename + '.csv')
        df = pd.DataFrame([data])
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        print(f"💾 Saved to: {filepath}")

if __name__ == "__main__":
    analyzer = HTMLAnalyzerComplete()
    folder_path = "html_input/whitelist"  # ปรับ path ตามโฟลเดอร์ไฟล์ HTML จริง
    analyzer.batch_analyze_html_folder(folder_path, list_type='white')
