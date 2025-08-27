import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
import os
import csv
import time
from urllib.parse import urlparse
import random
import whois
from datetime import datetime, date
import glob
from pathlib import Path

# Optional: Install mammoth for .docx support: pip install mammoth
try:
    import mammoth
    MAMMOTH_AVAILABLE = True
except ImportError:
    MAMMOTH_AVAILABLE = False
    print("⚠️  mammoth not installed - .docx files will be skipped. Install with: pip install mammoth")

class HTMLFileAnalyzer:
    def __init__(self):
        self.reference_date = date(2025, 1, 1)  # วันที่อ้างอิง 1 Jan 2025
    
    def read_html_from_file(self, filepath):
        """อ่าน HTML content จากไฟล์ต่างๆ"""
        try:
            file_path = Path(filepath)
            file_extension = file_path.suffix.lower()
            
            print(f"📄 Reading file: {filepath} (type: {file_extension})")
            
            if file_extension == '.docx':
                # อ่านไฟล์ .docx
                if not MAMMOTH_AVAILABLE:
                    print("❌ mammoth library not installed - cannot read .docx files")
                    print("   Install with: pip install mammoth")
                    return None
                    
                with open(filepath, 'rb') as docx_file:
                    result = mammoth.convert_to_html(docx_file)
                    html_content = result.value
                    if result.messages:
                        print(f"⚠️  Docx conversion messages: {result.messages}")
                
            elif file_extension in ['.html', '.htm', '.txt']:
                # อ่านไฟล์ HTML/TXT
                encodings = ['utf-8', 'utf-8-sig', 'iso-8859-1', 'windows-1252', 'cp1251']
                html_content = None
                
                for encoding in encodings:
                    try:
                        with open(filepath, 'r', encoding=encoding) as file:
                            html_content = file.read()
                        print(f"✅ Successfully read with encoding: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                
                if html_content is None:
                    print("❌ Failed to read file with any encoding")
                    return None
                    
            else:
                print(f"❌ Unsupported file type: {file_extension}")
                return None
            
            print(f"📊 HTML content length: {len(html_content)} characters")
            return html_content
            
        except Exception as e:
            print(f"❌ Error reading file {filepath}: {str(e)}")
            return None
    
    def extract_url_from_content(self, html_content):
        """พยายามดึง URL จาก HTML content"""
        try:
            # หา URL จาก canonical tag
            soup = BeautifulSoup(html_content, 'lxml')
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                return canonical.get('href')
            
            # หา URL จาก og:url
            og_url = soup.find('meta', property='og:url')
            if og_url and og_url.get('content'):
                return og_url.get('content')
            
            # หา URL จาก base tag
            base_tag = soup.find('base')
            if base_tag and base_tag.get('href'):
                return base_tag.get('href')
                
            # หาจาก title หรือ comment ที่อาจมี URL
            title = soup.find('title')
            if title and title.get_text():
                title_text = title.get_text()
                url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+\.[^\s<>"\']+'
                urls = re.findall(url_pattern, title_text)
                if urls:
                    url = urls[0]
                    if not url.startswith('http'):
                        url = 'https://' + url
                    return url
            
            # หาจาก comments ใน HTML
            comments = soup.find_all(string=lambda text: isinstance(text, str) and ('http' in text))
            for comment in comments:
                url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+\.[^\s<>"\']+'
                urls = re.findall(url_pattern, str(comment))
                if urls:
                    url = urls[0]
                    if not url.startswith('http'):
                        url = 'https://' + url
                    return url
                    
            return "unknown_url"
            
        except Exception as e:
            print(f"⚠️  Could not extract URL from content: {str(e)}")
            return "unknown_url"
    
    def get_domain_age(self, domain):
        """ตรวจสอบ domain age จาก WHOIS"""
        try:
            if domain == "unknown_domain" or not domain or '.' not in domain:
                return "suspect"
                
            print(f"🔍 Looking up WHOIS for: {domain}")
            
            # ลบ www. ถ้ามี
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # ลบ protocol ถ้ามี
            domain = re.sub(r'^https?://', '', domain)
            domain = domain.split('/')[0]  # เอาแค่ domain part
            
            # ดึงข้อมูล WHOIS
            domain_info = whois.whois(domain)
            
            # ตรวจสอบวันที่สร้าง
            creation_date = None
            
            if hasattr(domain_info, 'creation_date') and domain_info.creation_date:
                creation_date = domain_info.creation_date
                
                # ถ้าเป็น list ให้เอาตัวแรก
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                
                # แปลงเป็น date ถ้าเป็น datetime
                if isinstance(creation_date, datetime):
                    creation_date = creation_date.date()
            
            if creation_date:
                # คำนวณอายุ domain (จำนวนปี)
                age_days = (self.reference_date - creation_date).days
                age_years = age_days / 365.25  # ใช้ 365.25 เพื่อคิด leap year
                
                print(f"✅ Domain created: {creation_date}")
                print(f"📊 Domain age: {age_years:.2f} years")
                
                return f"{age_years:.2f} years"
            else:
                print("❌ Creation date not found in WHOIS")
                return "suspect"
                
        except Exception as e:
            print(f"❌ WHOIS lookup failed for {domain}: {str(e)}")
            return "suspect"
    
    def analyze_html_file(self, filepath, list_type='white'):
        """วิเคราะห์ไฟล์ HTML และสร้าง data string"""
        try:
            # อ่าน HTML content
            html_content = self.read_html_from_file(filepath)
            if not html_content:
                return None
            
            # Parse HTML
            soup = BeautifulSoup(html_content, 'lxml')
            
            # พยายามดึง URL จาก content
            extracted_url = self.extract_url_from_content(html_content)
            print(f"🔗 Extracted URL: {extracted_url}")
            
            # สร้าง data string
            result_string = self.create_line_by_line_string(soup, extracted_url)
            
            # เตรียมข้อมูลสำหรับ CSV
            result = {
                'data_string': result_string
            }
            
            # สร้างชื่อไฟล์สำหรับบันทึก
            file_stem = Path(filepath).stem
            output_filename = f"{file_stem}_analyzed"
            
            # บันทึกไฟล์
            self.save_to_csv(result, output_filename, list_type)
            
            print(f"✅ Analysis completed for: {filepath}")
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {filepath}: {str(e)}")
            return None
    
    def batch_analyze_files(self, folder_path, list_type='white', file_pattern='*'):
        """วิเคราะห์ไฟล์ทั้งหมดในโฟลเดอร์"""
        try:
            folder_path = Path(folder_path)
            if not folder_path.exists():
                print(f"❌ Folder not found: {folder_path}")
                return None
            
            # หาไฟล์ทั้งหมดที่รองรับ
            supported_extensions = ['*.html', '*.htm', '*.txt']
            if MAMMOTH_AVAILABLE:
                supported_extensions.append('*.docx')
            
            all_files = []
            
            for ext in supported_extensions:
                pattern = folder_path / ext
                files = list(glob.glob(str(pattern)))
                all_files.extend(files)
            
            # กรองตาม pattern ถ้ามี
            if file_pattern != '*':
                all_files = [f for f in all_files if file_pattern in Path(f).name]
            
            print(f"\n🚀 Starting batch analysis for {len(all_files)} files ({list_type}list)")
            print(f"📁 Source folder: {folder_path}")
            print(f"📁 Output folder: csv_train/{list_type}/")
            
            success_count = 0
            failed_count = 0
            
            for i, filepath in enumerate(all_files, 1):
                print(f"\n{'='*80}")
                print(f"📊 Progress: {i}/{len(all_files)} ({(i/len(all_files)*100):.1f}%)")
                print(f"📄 Processing: {Path(filepath).name}")
                
                try:
                    result = self.analyze_html_file(filepath, list_type)
                    if result:
                        success_count += 1
                        print(f"✅ Success: {Path(filepath).name}")
                    else:
                        failed_count += 1
                        print(f"❌ Failed: {Path(filepath).name}")
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Error with {Path(filepath).name}: {str(e)}")
                
                # Delay เล็กน้อยเพื่อให้ระบบพักฟื้น
                if i < len(all_files):
                    time.sleep(0.1)
            
            print(f"\n{'='*80}")
            print(f"🎉 Batch analysis completed!")
            print(f"✅ Successful: {success_count}")
            print(f"❌ Failed: {failed_count}")
            
            total_processed = success_count + failed_count
            if total_processed > 0:
                success_rate = (success_count / total_processed) * 100
                print(f"📊 Success rate: {success_rate:.1f}%")
            else:
                print("📊 No files were processed")
            
            return {
                'success_count': success_count,
                'failed_count': failed_count,
                'total': len(all_files)
            }
            
        except Exception as e:
            print(f"❌ Error in batch analysis: {str(e)}")
            return None
    
    def process_training_data_from_files(self, blacklist_folder='blacklist', whitelist_folder='whitelist'):
        """ประมวลผลข้อมูล training จากไฟล์ในโฟลเดอร์"""
        print("🎯 Starting training data processing from files...")
        
        results = {}
        
        # ตรวจสอบและประมวลผล blacklist folder
        if os.path.exists(blacklist_folder):
            print(f"\n🔴 Processing BLACKLIST files from: {blacklist_folder}")
            blacklist_results = self.batch_analyze_files(blacklist_folder, 'black')
            results['blacklist'] = blacklist_results
        else:
            print(f"⚠️  Blacklist folder not found: {blacklist_folder}")
            results['blacklist'] = {'success_count': 0, 'failed_count': 0, 'total': 0}
        
        # ตรวจสอบและประมวลผล whitelist folder
        if os.path.exists(whitelist_folder):
            print(f"\n🔵 Processing WHITELIST files from: {whitelist_folder}")
            whitelist_results = self.batch_analyze_files(whitelist_folder, 'white')
            results['whitelist'] = whitelist_results
        else:
            print(f"⚠️  Whitelist folder not found: {whitelist_folder}")
            results['whitelist'] = {'success_count': 0, 'failed_count': 0, 'total': 0}
        
        # สรุปผลรวม
        self.print_training_summary(results)
        
        return results
    
    def print_training_summary(self, results):
        """แสดงสรุปผลการประมวลผล"""
        blacklist_results = results.get('blacklist', {'success_count': 0, 'failed_count': 0, 'total': 0})
        whitelist_results = results.get('whitelist', {'success_count': 0, 'failed_count': 0, 'total': 0})
        
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING DATA PROCESSING COMPLETED!")
        print(f"{'='*80}")
        print(f"🔴 BLACKLIST Results:")
        print(f"   ✅ Successful: {blacklist_results['success_count']}")
        print(f"   ❌ Failed: {blacklist_results['failed_count']}")
        
        if blacklist_results['total'] > 0:
            blacklist_rate = (blacklist_results['success_count'] / blacklist_results['total'] * 100)
            print(f"   📊 Success rate: {blacklist_rate:.1f}%")
        else:
            print("   📊 Success rate: N/A (no files processed)")
        
        print(f"\n🔵 WHITELIST Results:")
        print(f"   ✅ Successful: {whitelist_results['success_count']}")
        print(f"   ❌ Failed: {whitelist_results['failed_count']}")
        
        if whitelist_results['total'] > 0:
            whitelist_rate = (whitelist_results['success_count'] / whitelist_results['total'] * 100)
            print(f"   📊 Success rate: {whitelist_rate:.1f}%")
        else:
            print("   📊 Success rate: N/A (no files processed)")
        
        total_success = blacklist_results['success_count'] + whitelist_results['success_count']
        total_processed = blacklist_results['total'] + whitelist_results['total']
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   🎯 Total files processed: {total_processed}")
        print(f"   ✅ Total successful: {total_success}")
        
        if total_processed > 0:
            overall_rate = (total_success / total_processed * 100)
            print(f"   📊 Overall success rate: {overall_rate:.1f}%")
        else:
            print("   📊 Overall success rate: N/A (no files processed)")
            
        print(f"   📁 Files saved in: csv_train/black/ and csv_train/white/")
    
    def create_line_by_line_string(self, soup, url):
        """สร้าง string แบบบรรทัดต่อบรรทัด"""
        lines = []
        
        # เพิ่ม URL และ Domain
        lines.append(f"URL: {url}")
        domain = self.extract_domain(url)
        lines.append(f"Domain: {domain}")
        
        # เพิ่ม Domain Age จาก WHOIS
        domain_age = self.get_domain_age(domain)
        lines.append(f"Domain_age: {domain_age}")
        
        # เพิ่ม Title
        title = self.extract_title(soup)
        lines.append(f"Title: {title}")
        
        # เพิ่ม Meta tags
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
        icon_type = self.analyze_icon_type(soup, url)
        lines.append(f"Icon_type: {icon_type}")
        
        # นับ script และ resources
        script_counts = self.count_all_scripts(soup, url)
        resource_counts = self.count_all_external_resources(soup, url)
        
        # เพิ่มข้อมูลการนับ
        lines.append(f"Number_of_redirect: 0")  # ไฟล์ local ไม่มี redirect
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {script_counts['total_scripts']}")
        lines.append(f"Number_of_src_script: {script_counts['src_scripts']}")
        lines.append(f"Number_of_inline_script: {script_counts['inline_scripts']}")
        lines.append(f"Number_of_external_script: {script_counts['external_scripts']}")
        lines.append(f"Number_of_external_resources: {resource_counts['total_external']}")
        
        # รวมเป็น string เดียว
        result_string = '\n'.join(lines)
        
        print(f"📊 Final statistics:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • Domain: {domain}")
        print(f"   • Domain age: {domain_age}")
        print(f"   • Meta tags: {len(meta_tags)}")
        print(f"   • Scripts: {script_counts['total_scripts']}")
        print(f"   • External resources: {resource_counts['total_external']}")
        print(f"   • String length: {len(result_string)} characters")
        
        return result_string
    
    # เมธอดเหล่านี้คงเดิมจากโค้ดต้นฉบับ
    def extract_domain(self, url):
        if url == "unknown_url":
            return "unknown_domain"
        try:
            return urlparse(url).netloc
        except:
            return "unknown_domain"
    
    def extract_title(self, soup):
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def clean_value(self, value):
        if value is None:
            return ""
        clean_val = str(value).replace('"', '').replace("'", '').replace('\n', ' ').replace('\r', ' ').strip()
        clean_val = re.sub(r'\s+', ' ', clean_val)
        return clean_val
    
    def analyze_icon_type(self, soup, base_url):
        """วิเคราะห์ icon type"""
        icon_selectors = [
            'link[rel*="icon"]',
            'link[rel="shortcut icon"]',
            'link[rel="apple-touch-icon"]',
            'link[rel="apple-touch-icon-precomposed"]',
            'link[rel="favicon"]', 
            'link[rel="mask-icon"]'
        ]
        
        if base_url == "unknown_url":
            return 'none'
            
        try:
            base_domain = urlparse(base_url).netloc
        except:
            return 'none'
            
        internal_count = 0
        external_count = 0
        
        for selector in icon_selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    try:
                        absolute_url = urljoin(base_url, href)
                        icon_domain = urlparse(absolute_url).netloc
                        
                        if icon_domain == base_domain or not icon_domain:
                            internal_count += 1
                        else:
                            external_count += 1
                    except:
                        continue
        
        if internal_count > 0 and external_count == 0:
            return 'internal'
        elif external_count > 0 and internal_count == 0:
            return 'external'
        elif internal_count > 0 and external_count > 0:
            return 'mixed'
        else:
            return 'none'
    
    def count_all_scripts(self, soup, base_url):
        """นับ script ทุกประเภท"""
        all_scripts = soup.find_all('script')
        
        if base_url == "unknown_url":
            base_domain = "unknown_domain"
        else:
            try:
                base_domain = urlparse(base_url).netloc
            except:
                base_domain = "unknown_domain"
        
        src_scripts = []
        inline_scripts = []
        external_scripts = []
        
        for script in all_scripts:
            src = script.get('src')
            
            if src:
                src_scripts.append(script)
                
                # ตรวจสอบ external
                try:
                    if src.startswith('http'):
                        script_domain = urlparse(src).netloc
                        if script_domain != base_domain:
                            external_scripts.append(script)
                    elif src.startswith('//'):
                        script_domain = urlparse('http:' + src).netloc
                        if script_domain != base_domain:
                            external_scripts.append(script)
                except:
                    pass
            else:
                inline_scripts.append(script)
        
        return {
            'total_scripts': len(all_scripts),
            'src_scripts': len(src_scripts),
            'inline_scripts': len(inline_scripts),
            'external_scripts': len(external_scripts)
        }
    
    def count_all_external_resources(self, soup, base_url):
        """นับ external resources"""
        if base_url == "unknown_url":
            base_domain = "unknown_domain"
        else:
            try:
                base_domain = urlparse(base_url).netloc
            except:
                base_domain = "unknown_domain"
        
        external_resources = []
        
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
            ('frame', 'src'),
            ('applet', 'archive'),
            ('area', 'href'),
            ('base', 'href')
        ]
        
        for tag_name, attr_name in resource_tags:
            for element in soup.find_all(tag_name):
                src = element.get(attr_name)
                if src:
                    try:
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
                    except:
                        continue
        
        return {
            'total_external': len(external_resources),
            'details': external_resources
        }
    
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
        folder_path = f'csv_train/{list_type}'
        os.makedirs(folder_path, exist_ok=True)
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = f'{folder_path}/{filename}'
        
        # บันทึกเป็น CSV
        df = pd.DataFrame([data])
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        
        print(f"💾 Saved to: {filepath}")
        
        # แสดงตัวอย่าง string
        preview = data['data_string'][:200] + "..." if len(data['data_string']) > 200 else data['data_string']
        print(f"📋 Data preview: {preview}")

# การใช้งาน
if __name__ == "__main__":
    analyzer = HTMLFileAnalyzer()
    
    # ตัวอย่างการใช้งาน
    print("🎯 HTML File Analyzer")
    print("=" * 50)
    
    # วิธีที่ 1: วิเคราะห์ไฟล์เดียว
    # analyzer.analyze_html_file('path/to/your/file.html', 'white')
    
    # วิธีที่ 2: วิเคราะห์ไฟล์ทั้งหมดในโฟลเดอร์
    # analyzer.batch_analyze_files('path/to/html/files', 'white')
    
    # วิธีที่ 3: ประมวลผลข้อมูล training จากโฟลเดอร์ blacklist และ whitelist
    results = analyzer.process_training_data_from_files('training/blacktrain', 'training/whitetrain')
    
    print(f"\n🎉 Analysis completed! Check csv_training folder for results.")