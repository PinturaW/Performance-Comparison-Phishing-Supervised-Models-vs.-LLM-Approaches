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
    
    def convert_drive_link_to_direct(self, drive_url):
        """แปลง Google Drive share link เป็น direct download link"""
        try:
            if '/file/d/' in drive_url:
                file_id = drive_url.split('/file/d/')[1].split('/')[0]
                direct_url = f"https://drive.google.com/uc?id={file_id}&export=download"
                print(f"✅ Converted to direct link")
                return direct_url
            else:
                print(f"❌ Invalid Google Drive link format")
                return None
        except Exception as e:
            print(f"❌ Error converting Drive link: {str(e)}")
            return None
    
    def load_html_source_from_drive(self, drive_url):
        """ดาวน์โหลด HTML source จาก Google Drive"""
        try:
            print(f"📥 Downloading HTML source from Drive...")
            
            # แปลงเป็น direct download link
            direct_url = self.convert_drive_link_to_direct(drive_url)
            if not direct_url:
                return None
            
            # ดาวน์โหลดเนื้อหา
            response = requests.get(direct_url, timeout=30)
            response.raise_for_status()
            
            html_content = response.text
            print(f"✅ Successfully downloaded HTML source: {len(html_content)} characters")
            return html_content
            
        except Exception as e:
            print(f"❌ Error downloading HTML from Drive: {str(e)}")
            return None
    
    def analyze_html_source_from_drive(self, drive_url, output_filename=None, list_type='white'):
        """วิเคราะห์ HTML source จาก Google Drive และสร้าง data string"""
        try:
            print(f"🔍 Analyzing HTML source from Google Drive")
            print(f"🔗 Drive URL: {drive_url}")
            
            # ดาวน์โหลด HTML content
            html_content = self.load_html_source_from_drive(drive_url)
            if not html_content:
                return None
            
            # สร้าง BeautifulSoup object
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # ใช้ชื่อไฟล์จาก Drive URL ถ้าไม่ได้ระบุ
            if not output_filename:
                try:
                    file_id = drive_url.split('/file/d/')[1].split('/')[0]
                    output_filename = f"drive_html_{file_id[:8]}"
                except:
                    output_filename = f"drive_html_{random.randint(1000, 9999)}"
            
            # สร้าง data string
            result_string = self.create_line_by_line_string_from_html_source(soup, drive_url)
            
            # เตรียมข้อมูลสำหรับ CSV
            result = {
                'data_string': result_string
            }
            
            # บันทึกไฟล์
            self.save_to_csv(result, output_filename, list_type)
            
            print(f"✅ HTML source analysis completed!")
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing HTML source from Drive: {str(e)}")
            return None
    
    def create_line_by_line_string_from_html_source(self, soup, drive_url):
        """สร้าง data string แบบบรรทัดต่อบรรทัด จาก HTML source"""
        lines = []
        
        # เพิ่มข้อมูลพื้นฐาน
        lines.append(f"Source: Google_Drive_HTML")
        lines.append(f"Drive_URL: {drive_url}")
        
        # ลองหา URL จาก canonical link หรือ meta tags
        url = self.extract_url_from_html(soup)
        if url:
            lines.append(f"URL: {url}")
            domain = self.extract_domain(url)
            lines.append(f"Domain: {domain}")
            
            # ลอง domain age (ถ้าหา URL ได้)
            domain_age = self.get_domain_age(domain)
            lines.append(f"Domain_age: {domain_age}")
        else:
            lines.append(f"URL: html_source_analysis")
            lines.append(f"Domain: html_source_analysis")
            lines.append(f"Domain_age: html_source_analysis")
        
        # เพิ่ม Title
        title = self.extract_title(soup)
        lines.append(f"Title: {title}")
        
        # เพิ่ม Meta tags แต่ละตัวเป็นบรรทัด
        meta_tags = soup.find_all('meta')
        print(f"\n📋 Processing {len(meta_tags)} meta tags:")
        
        for i, meta in enumerate(meta_tags, 1):
            if meta.get('name'):
                name_val = self.clean_value(meta.get('name'))
                content_val = self.clean_value(meta.get('content', ''))
                line = f"Meta_{name_val}: {content_val}"
                lines.append(line)
                
            elif meta.get('property'):
                prop_val = self.clean_value(meta.get('property'))
                content_val = self.clean_value(meta.get('content', ''))
                line = f"Meta_{prop_val}: {content_val}"
                lines.append(line)
                
            elif meta.get('http-equiv'):
                equiv_val = self.clean_value(meta.get('http-equiv'))
                content_val = self.clean_value(meta.get('content', ''))
                line = f"Meta_{equiv_val}: {content_val}"
                lines.append(line)
                
            elif meta.get('charset'):
                charset_val = self.clean_value(meta.get('charset'))
                line = f"Meta_charset: {charset_val}"
                lines.append(line)
                
            elif meta.get('itemprop'):
                item_val = self.clean_value(meta.get('itemprop'))
                content_val = self.clean_value(meta.get('content', ''))
                line = f"Meta_{item_val}: {content_val}"
                lines.append(line)
                
            elif meta.get('id'):
                id_val = self.clean_value(meta.get('id'))
                # สำหรับ meta id="config" ที่มีข้อมูลเยอะ
                attrs_list = []
                for k, v in meta.attrs.items():
                    if k != 'id':  # ไม่เอา id ซ้ำ
                        clean_k = self.clean_value(k)
                        clean_v = self.clean_value(v)
                        if clean_v:
                            attrs_list.append(f"{clean_k}={clean_v}")
                
                if attrs_list:
                    line = f"Meta_{id_val}: {','.join(attrs_list)}"
                    lines.append(line)
                
            else:
                # สำหรับ attributes อื่นๆ
                attrs_list = []
                for k, v in meta.attrs.items():
                    clean_k = self.clean_value(k)
                    clean_v = self.clean_value(v)
                    if clean_v:
                        attrs_list.append(f"{clean_k}={clean_v}")
                    else:
                        attrs_list.append(clean_k)
                
                if attrs_list:
                    line = f"Meta_{','.join(attrs_list)}:"
                    lines.append(line)
        
        # เพิ่ม Icon type
        icon_type = self.analyze_icon_type_from_html(soup, url)
        lines.append(f"Icon_type: {icon_type}")
        
        # นับ script และ resources
        script_counts = self.count_all_scripts_from_html(soup, url)
        resource_counts = self.count_all_external_resources_from_html(soup, url)
        
        # แยกเป็นบรรทัด Number of ต่างๆ
        lines.append(f"Number_of_redirect: 0")  # HTML source ไม่มี redirect
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {script_counts['total_scripts']}")
        lines.append(f"Number_of_src_script: {script_counts['src_scripts']}")
        lines.append(f"Number_of_inline_script: {script_counts['inline_scripts']}")
        lines.append(f"Number_of_external_script: {script_counts['external_scripts']}")
        lines.append(f"Number_of_external_resources: {resource_counts['total_external']}")
        
        # รวมเป็น string เดียวด้วย newline
        result_string = '\n'.join(lines)
        
        print(f"📊 Final statistics:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • Meta tags: {len(meta_tags)}")
        print(f"   • Scripts: {script_counts['total_scripts']}")
        print(f"   • External resources: {resource_counts['total_external']}")
        print(f"   • String length: {len(result_string)} characters")
        
        return result_string
    
    def extract_url_from_html(self, soup):
        """พยายามหา URL จาก HTML"""
        try:
            # หาจาก canonical link
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                return canonical.get('href')
            
            # หาจาก og:url
            og_url = soup.find('meta', property='og:url')
            if og_url and og_url.get('content'):
                return og_url.get('content')
            
            # หาจาก base tag
            base = soup.find('base')
            if base and base.get('href'):
                return base.get('href')
            
            return None
        except:
            return None
    
    def analyze_icon_type_from_html(self, soup, base_url):
        """วิเคราะห์ icon type จาก HTML source"""
        icon_selectors = [
            'link[rel*="icon"]',
            'link[rel="shortcut icon"]',
            'link[rel="apple-touch-icon"]',
            'link[rel="apple-touch-icon-precomposed"]',
            'link[rel="favicon"]',
            'link[rel="mask-icon"]'
        ]
        
        if not base_url:
            # ถ้าไม่มี base URL ให้ตรวจดูแค่ว่ามี icon หรือไม่
            icon_count = 0
            for selector in icon_selectors:
                icons = soup.select(selector)
                icon_count += len(icons)
            
            return 'internal' if icon_count > 0 else 'none'
        
        # ถ้ามี base URL ให้ตรวจ internal/external
        base_domain = urlparse(base_url).netloc
        internal_count = 0
        external_count = 0
        
        for selector in icon_selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    if href.startswith('http'):
                        icon_domain = urlparse(href).netloc
                        if icon_domain == base_domain:
                            internal_count += 1
                        else:
                            external_count += 1
                    else:
                        internal_count += 1  # relative path = internal
        
        if internal_count > 0 and external_count == 0:
            return 'internal'
        elif external_count > 0 and internal_count == 0:
            return 'external'
        elif internal_count > 0 and external_count > 0:
            return 'mixed'
        else:
            return 'none'
    
    def count_all_scripts_from_html(self, soup, base_url):
        """นับ script จาก HTML source"""
        all_scripts = soup.find_all('script')
        
        src_scripts = []
        inline_scripts = []
        external_scripts = []
        
        for script in all_scripts:
            src = script.get('src')
            
            if src:
                src_scripts.append(script)
                
                # ตรวจสอบ external
                if src.startswith('http') or src.startswith('//'):
                    if base_url:
                        base_domain = urlparse(base_url).netloc
                        if src.startswith('//'):
                            script_domain = urlparse('http:' + src).netloc
                        else:
                            script_domain = urlparse(src).netloc
                        
                        if script_domain != base_domain:
                            external_scripts.append(script)
                    else:
                        # ถ้าไม่มี base URL ให้ถือว่า http/https เป็น external
                        external_scripts.append(script)
            else:
                inline_scripts.append(script)
        
        return {
            'total_scripts': len(all_scripts),
            'src_scripts': len(src_scripts),
            'inline_scripts': len(inline_scripts),
            'external_scripts': len(external_scripts)
        }
    
    def count_all_external_resources_from_html(self, soup, base_url):
        """นับ external resources จาก HTML source"""
        resource_tags = [
            ('script', 'src'),
            ('link', 'href'),
            ('img', 'src'),
            ('iframe', 'src'),
            ('embed', 'src'),
            ('object', 'data'),
            ('source', 'src'),
            ('track', 'src'),
            ('video', 'src'),
            ('audio', 'src'),
            ('input', 'src'),
            ('frame', 'src')
        ]
        
        external_resources = []
        
        if not base_url:
            # ถ้าไม่มี base URL ให้นับแค่ที่ขึ้นต้นด้วย http
            for tag_name, attr_name in resource_tags:
                for element in soup.find_all(tag_name):
                    src = element.get(attr_name)
                    if src and (src.startswith('http') or src.startswith('//')):
                        external_resources.append({
                            'tag': tag_name,
                            'attr': attr_name,
                            'url': src
                        })
        else:
            # ถ้ามี base URL ให้เปรียบเทียบ domain
            base_domain = urlparse(base_url).netloc
            
            for tag_name, attr_name in resource_tags:
                for element in soup.find_all(tag_name):
                    src = element.get(attr_name)
                    if src:
                        is_external = False
                        
                        if src.startswith('http'):
                            src_domain = urlparse(src).netloc
                            if src_domain != base_domain:
                                is_external = True
                        elif src.startswith('//'):
                            src_domain = urlparse('http:' + src).netloc
                            if src_domain != base_domain:
                                is_external = True
                        
                        if is_external:
                            external_resources.append({
                                'tag': tag_name,
                                'attr': attr_name,
                                'url': src
                            })
        
        return {
            'total_external': len(external_resources),
            'details': external_resources
        }
    
    def batch_analyze_html_sources_from_drive(self, drive_urls, list_type='white'):
        """วิเคราะห์ HTML sources หลายไฟล์จาก Google Drive"""
        print(f"\n🚀 Starting batch HTML source analysis for {len(drive_urls)} files ({list_type})")
        print(f"📁 Files will be saved in: csv_training/{list_type}/")
        
        success_count = 0
        failed_count = 0
        
        for i, drive_url in enumerate(drive_urls, 1):
            print(f"\n{'='*60}")
            print(f"📊 Progress: {i}/{len(drive_urls)} ({(i/len(drive_urls)*100):.1f}%)")
            
            try:
                # สร้างชื่อไฟล์
                try:
                    file_id = drive_url.split('/file/d/')[1].split('/')[0]
                    filename = f"html_source_{file_id[:8]}_{i}"
                except:
                    filename = f"html_source_{i}_{random.randint(1000, 9999)}"
                
                result = self.analyze_html_source_from_drive(drive_url, filename, list_type)
                if result:
                    success_count += 1
                    print(f"✅ Success: {drive_url}")
                else:
                    failed_count += 1
                    print(f"❌ Failed: {drive_url}")
            except Exception as e:
                failed_count += 1
                print(f"❌ Error with {drive_url}: {str(e)}")
            
            # Delay เพื่อไม่ให้โหลด Drive เร็วเกินไป
            if i < len(drive_urls):
                time.sleep(1)
        
        print(f"\n{'='*60}")
        print(f"🎉 Batch HTML source analysis completed!")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        total_processed = success_count + failed_count
        if total_processed > 0:
            success_rate = (success_count / total_processed) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")
        
        return {
            'success_count': success_count,
            'failed_count': failed_count,
            'total': len(drive_urls)
        }
    
    # ฟังก์ชันสำคัญอื่นๆ
    def get_domain_age(self, domain):
        """ตรวจสอบ domain age จาก WHOIS"""
        try:
            print(f"🔍 Looking up WHOIS for: {domain}")
            
            if domain.startswith('www.'):
                domain = domain[4:]
            
            domain_info = whois.whois(domain)
            creation_date = None
            
            if hasattr(domain_info, 'creation_date') and domain_info.creation_date:
                creation_date = domain_info.creation_date
                
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                if isinstance(creation_date, datetime):
                    creation_date = creation_date.date()
            
            if creation_date:
                age_days = (self.reference_date - creation_date).days
                age_years = age_days / 365.25
                
                print(f"✅ Domain created: {creation_date}")
                print(f"📊 Domain age: {age_years:.2f} years")
                
                return f"{age_years:.2f} years"
            else:
                print("❌ Creation date not found in WHOIS")
                return "suspect"
                
        except Exception as e:
            print(f"❌ WHOIS lookup failed for {domain}: {str(e)}")
            return "suspect"
    
    def extract_domain(self, url):
        """แยก domain จาก URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except Exception:
            return "unknown"
    
    def extract_title(self, soup):
        """แยก title จาก HTML"""
        try:
            title_tag = soup.find('title')
            if title_tag:
                return self.clean_value(title_tag.get_text())
            return "no_title"
        except Exception:
            return "no_title"
    
    def clean_value(self, value):
        """ทำความสะอาดค่าสำหรับ CSV"""
        if value is None:
            return ""
        
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        
        value = str(value).strip()
        value = value.replace('\n', ' ').replace('\r', ' ')
        value = re.sub(r'\s+', ' ', value)
        value = value.replace('"', '').replace("'", '')
        
        return value
    
    def count_popups(self, soup):
        """นับ popup patterns"""
        popup_patterns = [
            r'window\.open\s*\(',
            r'alert\s*\(',
            r'confirm\s*\(',
            r'prompt\s*\(',
            r'\.modal\s*\(',
            r'showModal',
            r'openPopup',
            r'\.popup\s*\(',
            r'\.dialog\s*\('
        ]
        
        popup_count = 0
        for script in soup.find_all('script'):
            if script.string:
                for pattern in popup_patterns:
                    matches = re.findall(pattern, script.string, re.IGNORECASE)
                    popup_count += len(matches)
        
        return popup_count
    
    def save_to_csv(self, data, filename, list_type):
        """บันทึกเป็น CSV"""
        try:
            folder_path = f'csv_training/{list_type}'
            os.makedirs(folder_path, exist_ok=True)
            
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            filepath = f'{folder_path}/{filename}'
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(['data_string'])  # header
                writer.writerow([data['data_string']])
            
            print(f"💾 Saved to: {filepath}")
            
            # แสดงตัวอย่าง string
            preview = data['data_string'][:200] + "..." if len(data['data_string']) > 200 else data['data_string']
            print(f"📋 Data preview: {preview}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error saving CSV: {str(e)}")
            return False

# การใช้งาน
if __name__ == "__main__":
    analyzer = HTMLAnalyzerComplete()
    
    # ตัวอย่างการใช้งาน - วิเคราะห์ HTML source ไฟล์เดียว
    print("🎯 HTML Source Analysis from Google Drive")
    
    # ใส่ Google Drive link ของไฟล์ .txt ที่มี HTML source
    html_source_drive_url = "https://drive.google.com/drive/u/0/folders/1BoNJeGgBFBcHrlGBaznC4q1JXvIk6c_f"
    
    # วิเคราะห์ไฟล์เดียว
    result = analyzer.analyze_html_source_from_drive(
        html_source_drive_url, 
        "linkedin_example", 
        "white"
    )
    
    if result:
        print(f"\n🎉 Single file analysis completed!")
    
    # หรือสำหรับหลายไฟล์
    """
    html_source_urls = [
        "https://drive.google.com/file/d/FILE_ID_1/view?usp=sharing",
        "https://drive.google.com/file/d/FILE_ID_2/view?usp=sharing",
        "https://drive.google.com/file/d/FILE_ID_3/view?usp=sharing"
    ]
    
    results = analyzer.batch_analyze_html_sources_from_drive(html_source_urls, 'white')
    print(f"\n🎉 Batch analysis completed!")
    """
