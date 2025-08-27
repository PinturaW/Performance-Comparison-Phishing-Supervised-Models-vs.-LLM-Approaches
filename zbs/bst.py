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

class HTMLAnalyzerComplete:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.redirects_count = 0
    
    def load_urls_from_file(self, filename):
        """อ่าน URLs จากไฟล์"""
        try:
            # ใช้ Python standard file reading แทน window.fs.readFile
            with open(filename, 'r', encoding='utf-8') as file:
                content = file.read()
                urls = [url.strip() for url in content.split('\n') if url.strip()]
                return urls
        except FileNotFoundError:
            print(f"❌ Error: File '{filename}' not found")
            return []
        except Exception as e:
            print(f"❌ Error reading file {filename}: {str(e)}")
            return []
    
    def generate_filename_from_url(self, url):
        """สร้างชื่อไฟล์จาก URL"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            
            # ลบ www. หากมี
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # แทนที่ตัวอักษรที่ไม่ใช่ตัวอักษรและตัวเลขด้วย underscore
            filename = re.sub(r'[^a-zA-Z0-9]', '_', domain)
            
            # ลบ underscore ที่ซ้ำติดกัน
            filename = re.sub(r'_+', '_', filename)
            
            # ลบ underscore ที่ขึ้นต้นและลงท้าย
            filename = filename.strip('_')
            
            # ถ้าชื่อยาวเกินไป ให้ตัดให้เหลือ 50 ตัวอักษร
            if len(filename) > 50:
                filename = filename[:50]
            
            return filename
        except Exception as e:
            print(f"❌ Error generating filename for {url}: {str(e)}")
            # สำรองใช้เลขสุ่ม
            return f"unknown_{random.randint(1000, 9999)}"
    
    def analyze_single_url(self, url, filename, list_type='white'):
        """วิเคราะห์ URL และแสดงผลแบบบรรทัดต่อบรรทัด"""
        try:
            print(f"🔍 Analyzing: {url}")
            
            response = self.get_html_with_redirect_count(url)
            if not response:
                print(f"❌ Failed to fetch {url}")
                return None
            
            soup = BeautifulSoup(response.text, 'lxml')
            print(f"📄 HTML content length: {len(response.text)} characters")
            
            # สร้าง string แบบบรรทัดต่อบรรทัด
            result_string = self.create_line_by_line_string(soup, response.url)
            
            # เตรียมข้อมูลสำหรับ CSV
            result = {
                'data_string': result_string
            }
            
            # บันทึกไฟล์
            self.save_to_csv(result, filename, list_type)
            
            print(f"✅ Analysis completed!")
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {url}: {str(e)}")
            return None
    
    def batch_analyze_urls(self, urls, list_type='white', delay=1):
        """วิเคราะห์ URLs เป็นกลุ่ม"""
        print(f"\n🚀 Starting batch analysis for {len(urls)} URLs ({list_type}list)")
        print(f"📁 Files will be saved in: csv_training/{list_type}/")
        
        success_count = 0
        failed_count = 0
        
        for i, url in enumerate(urls, 1):
            print(f"\n{'='*60}")
            print(f"📊 Progress: {i}/{len(urls)} ({(i/len(urls)*100):.1f}%)")
            
            # สร้างชื่อไฟล์
            filename = self.generate_filename_from_url(url)
            
            try:
                result = self.analyze_single_url(url, filename, list_type)
                if result:
                    success_count += 1
                    print(f"✅ Success: {url}")
                else:
                    failed_count += 1
                    print(f"❌ Failed: {url}")
            except Exception as e:
                failed_count += 1
                print(f"❌ Error with {url}: {str(e)}")
            
            # Delay เพื่อไม่ให้ส่ง request เร็วเกินไป
            if i < len(urls):  # ไม่ต้อง delay หลังจาก URL สุดท้าย
                time.sleep(delay)
        
        print(f"\n{'='*60}")
        print(f"🎉 Batch analysis completed!")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        # Fixed division by zero error
        total_processed = success_count + failed_count
        if total_processed > 0:
            success_rate = (success_count / total_processed) * 100
            print(f"📊 Success rate: {success_rate:.1f}%")
        else:
            print("📊 No URLs were processed")
        
        return {
            'success_count': success_count,
            'failed_count': failed_count,
            'total': len(urls)
        }
    
    def process_training_data(self):
        """ประมวลผลข้อมูล training ทั้ง blacklist และ whitelist"""
        print("🎯 Starting training data processing...")
        
        # อ่าน blacklist URLs
        print("\n📋 Loading blacklist URLs...")
        blacklist_urls = self.load_urls_from_file('paste-2.txt')
        print(f"📊 Found {len(blacklist_urls)} blacklist URLs")
        
        # อ่าน whitelist URLs  
        print("\n📋 Loading whitelist URLs...")
        whitelist_urls = self.load_urls_from_file('paste-3.txt')
        print(f"📊 Found {len(whitelist_urls)} whitelist URLs")
        
        # จำกัดจำนวน URLs ตามที่ต้องการ (400 URLs สำหรับแต่ละประเภท)
        blacklist_urls = blacklist_urls[:400]
        whitelist_urls = whitelist_urls[:400]
        
        print(f"\n🎯 Processing {len(blacklist_urls)} blacklist URLs and {len(whitelist_urls)} whitelist URLs")
        
        # ประมวลผล blacklist
        print(f"\n🔴 Processing BLACKLIST URLs...")
        blacklist_results = self.batch_analyze_urls(blacklist_urls, 'black', delay=2)
        
        # ประมวลผล whitelist
        print(f"\n🔵 Processing WHITELIST URLs...")
        whitelist_results = self.batch_analyze_urls(whitelist_urls, 'white', delay=2)
        
        # สรุปผลรวม
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING DATA PROCESSING COMPLETED!")
        print(f"{'='*80}")
        print(f"🔴 BLACKLIST Results:")
        print(f"   ✅ Successful: {blacklist_results['success_count']}")
        print(f"   ❌ Failed: {blacklist_results['failed_count']}")
        
        # Fixed division by zero error for blacklist
        if blacklist_results['total'] > 0:
            blacklist_rate = (blacklist_results['success_count'] / blacklist_results['total'] * 100)
            print(f"   📊 Success rate: {blacklist_rate:.1f}%")
        else:
            print("   📊 Success rate: N/A (no URLs processed)")
        
        print(f"\n🔵 WHITELIST Results:")
        print(f"   ✅ Successful: {whitelist_results['success_count']}")
        print(f"   ❌ Failed: {whitelist_results['failed_count']}")
        
        # Fixed division by zero error for whitelist
        if whitelist_results['total'] > 0:
            whitelist_rate = (whitelist_results['success_count'] / whitelist_results['total'] * 100)
            print(f"   📊 Success rate: {whitelist_rate:.1f}%")
        else:
            print("   📊 Success rate: N/A (no URLs processed)")
        
        total_success = blacklist_results['success_count'] + whitelist_results['success_count']
        total_processed = blacklist_results['total'] + whitelist_results['total']
        
        print(f"\n📊 OVERALL STATISTICS:")
        print(f"   🎯 Total URLs processed: {total_processed}")
        print(f"   ✅ Total successful: {total_success}")
        
        # Fixed division by zero error for overall
        if total_processed > 0:
            overall_rate = (total_success / total_processed * 100)
            print(f"   📊 Overall success rate: {overall_rate:.1f}%")
        else:
            print("   📊 Overall success rate: N/A (no URLs processed)")
            
        print(f"   📁 Files saved in: csv_training/black/ and csv_training/white/")
        
        return {
            'blacklist': blacklist_results,
            'whitelist': whitelist_results,
            'total_success': total_success,
            'total_processed': total_processed
        }
    
    def create_line_by_line_string(self, soup, url):
        """สร้าง string แบบบรรทัดต่อบรรทัด ครบถ้วนทุกข้อมูล"""
        lines = []
        
        # เพิ่ม URL และ Domain
        lines.append(f"URL: {url}")
        lines.append(f"Domain: {self.extract_domain(url)}")
        
        # เพิ่ม Title
        title = self.extract_title(soup)
        lines.append(f"Title: {title}")
        
        # เพิ่ม Meta tags แต่ละตัวเป็นบรรทัด - ไม่ตัดเลย
        meta_tags = soup.find_all('meta')
        
        print(f"\n📋 Processing all {len(meta_tags)} meta tags:")
        
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
        
        # นับ script และ resources ตามที่กำหนด
        script_counts = self.count_all_scripts(soup, url)
        resource_counts = self.count_all_external_resources(soup, url)
        
        # แยกเป็นบรรทัด Number of ต่างๆ
        lines.append(f"Number_of_redirect: {self.redirects_count}")
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
        print(f"   • Scripts: {script_counts['total_scripts']} (Src: {script_counts['src_scripts']}, Inline: {script_counts['inline_scripts']}, External: {script_counts['external_scripts']})")
        print(f"   • External resources: {resource_counts['total_external']}")
        print(f"   • String length: {len(result_string)} characters")
        
        return result_string
    
    def get_html_with_redirect_count(self, url):
        """ดึง HTML และนับ redirect"""
        self.redirects_count = 0
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            response = self.session.get(url, timeout=30, allow_redirects=False, headers=headers)
            
            while response.status_code in [301, 302, 303, 307, 308]:
                self.redirects_count += 1
                redirect_url = response.headers.get('Location')
                if redirect_url:
                    if not redirect_url.startswith('http'):
                        redirect_url = urljoin(url, redirect_url)
                    response = self.session.get(redirect_url, timeout=30, allow_redirects=False, headers=headers)
                else:
                    break
                
                # ป้องกัน infinite redirect
                if self.redirects_count > 10:
                    print("⚠️ Too many redirects, stopping...")
                    break
            
            if response.status_code == 200:
                return response
            else:
                response.raise_for_status()
                
        except requests.RequestException as e:
            print(f"Failed to fetch {url}: {str(e)}")
            return None
    
    def extract_domain(self, url):
        return urlparse(url).netloc
    
    def extract_title(self, soup):
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else ""
    
    def clean_value(self, value):
        """ทำความสะอาดค่าโดยไม่ตัดข้อมูล"""
        if value is None:
            return ""
        # ลบเฉพาะตัวอักษรที่อาจทำให้ CSV เสีย
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
        
        base_domain = urlparse(base_url).netloc
        internal_count = 0
        external_count = 0
        
        for selector in icon_selectors:
            for link in soup.select(selector):
                href = link.get('href')
                if href:
                    absolute_url = urljoin(base_url, href)
                    icon_domain = urlparse(absolute_url).netloc
                    
                    if icon_domain == base_domain or not icon_domain:
                        internal_count += 1
                    else:
                        external_count += 1
        
        if internal_count > 0 and external_count == 0:
            return 'internal'
        elif external_count > 0 and internal_count == 0:
            return 'external'
        elif internal_count > 0 and external_count > 0:
            return 'mixed'
        else:
            return 'none'
    
    def count_all_scripts(self, soup, base_url):
        """นับ script ทุกประเภทตามคำจำกัดความที่ชัดเจน"""
        all_scripts = soup.find_all('script')
        base_domain = urlparse(base_url).netloc
        
        src_scripts = []
        inline_scripts = []
        external_scripts = []
        
        for i, script in enumerate(all_scripts, 1):
            src = script.get('src')
            
            if src:
                # Script ที่มี src attribute
                src_scripts.append(script)
                
                # ตรวจสอบว่าเป็น external หรือไม่
                if src.startswith('http'):
                    # URL เต็ม
                    script_domain = urlparse(src).netloc
                    if script_domain != base_domain:
                        external_scripts.append(script)
                elif src.startswith('//'):
                    # Protocol-relative URL
                    script_domain = urlparse('http:' + src).netloc
                    if script_domain != base_domain:
                        external_scripts.append(script)
            else:
                # Script ที่ไม่มี src (inline)
                inline_scripts.append(script)
        
        result = {
            'total_scripts': len(all_scripts),
            'src_scripts': len(src_scripts),
            'inline_scripts': len(inline_scripts),
            'external_scripts': len(external_scripts)
        }
        
        return result
    
    def count_all_external_resources(self, soup, base_url):
        """นับ external resources ทุกประเภท"""
        base_domain = urlparse(base_url).netloc
        external_resources = []
        
        # รายการ tags และ attributes ที่ต้องตรวจสอบ
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
            ('input', 'src'),  # สำหรับ input type="image"
            ('frame', 'src'),
            ('applet', 'archive'),
            ('area', 'href'),
            ('base', 'href')
        ]
        
        external_by_type = {}
        
        for tag_name, attr_name in resource_tags:
            count = 0
            for element in soup.find_all(tag_name):
                src = element.get(attr_name)
                if src:
                    # ตรวจสอบว่าเป็น external หรือไม่
                    is_external = False
                    
                    if src.startswith('http'):
                        # URL เต็ม
                        src_domain = urlparse(src).netloc
                        if src_domain != base_domain:
                            is_external = True
                    elif src.startswith('//'):
                        # Protocol-relative URL
                        src_domain = urlparse('http:' + src).netloc
                        if src_domain != base_domain:
                            is_external = True
                    
                    if is_external:
                        external_resources.append({
                            'tag': tag_name,
                            'attr': attr_name,
                            'url': src
                        })
                        count += 1
            
            if count > 0:
                external_by_type[tag_name] = count
        
        result = {
            'total_external': len(external_resources),
            'by_type': external_by_type,
            'details': external_resources
        }
        
        return result
    
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
        """บันทึกเป็น CSV โดยไม่ตัดข้อมูล"""
        folder_path = f'csv_training/{list_type}'
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
        
        # ตรวจสอบความสมบูรณ์ของข้อมูล
        lines = data['data_string'].split('\n')
        print(f"📊 Lines: {len(lines)}, Length: {len(data['data_string'])} chars")

# การใช้งาน
if __name__ == "__main__":
    analyzer = HTMLAnalyzerComplete()
    
    # เรียกใช้งานประมวลผลข้อมูล training
    print("🎯 Starting Training Data Processing...")
    results = analyzer.process_training_data()
    
    print(f"\n🎉 All done! Check your csv_training folder for results.")