import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re
import pandas as pd
import os
import csv

class HTMLAnalyzerComplete:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.redirects_count = 0
    
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
        
        # นับ script และ external resources แยกกัน
        script_counts = self.count_all_scripts(soup, url)
        external_non_script_counts = self.count_external_non_script_resources(soup, url)
        
        # แยกเป็นบรรทัด Number of ต่างๆ ตามที่คุณต้องการ
        lines.append(f"Number_of_redirect: {self.redirects_count}")
        lines.append(f"Number_of_popup: {self.count_popups(soup)}")
        lines.append(f"Number_of_script: {script_counts['total_scripts']}")
        lines.append(f"Number_of_src_script: {script_counts['src_scripts']}")
        lines.append(f"Number_of_inline_script: {script_counts['inline_scripts']}")
        lines.append(f"Number_of_external_src_script: {script_counts['external_scripts']}")  # แยกออกมาชัดเจน
        lines.append(f"Number_of_external_src: {external_non_script_counts['total_external']}")  # resource อื่นที่ไม่ใช่ script
        
        # รวมเป็น string เดียวด้วย newline
        result_string = '\n'.join(lines)
        
        print(f"📊 Final statistics:")
        print(f"   • Total lines: {len(lines)}")
        print(f"   • Meta tags: {len(meta_tags)}")
        print(f"   • Scripts: {script_counts['total_scripts']} (Src: {script_counts['src_scripts']}, Inline: {script_counts['inline_scripts']}, External: {script_counts['external_scripts']})")
        print(f"   • External non-script resources: {external_non_script_counts['total_external']}")
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
        
        print(f"\n📊 Complete Script Analysis:")
        print(f"   • Total script tags found: {len(all_scripts)}")
        
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
                        print(f"   {i:2d}. EXTERNAL SRC: {src}")
                    else:
                        print(f"   {i:2d}. INTERNAL SRC: {src}")
                elif src.startswith('//'):
                    # Protocol-relative URL
                    script_domain = urlparse('http:' + src).netloc
                    if script_domain != base_domain:
                        external_scripts.append(script)
                        print(f"   {i:2d}. EXTERNAL SRC: {src}")
                    else:
                        print(f"   {i:2d}. INTERNAL SRC: {src}")
                else:
                    # Relative URL - ถือว่าเป็น internal
                    print(f"   {i:2d}. INTERNAL SRC: {src}")
            else:
                # Script ที่ไม่มี src (inline)
                inline_scripts.append(script)
                content_preview = ""
                if script.string:
                    content_preview = script.string.strip()[:50]
                    if len(script.string.strip()) > 50:
                        content_preview += "..."
                print(f"   {i:2d}. INLINE: {content_preview}")
        
        result = {
            'total_scripts': len(all_scripts),
            'src_scripts': len(src_scripts),
            'inline_scripts': len(inline_scripts),
            'external_scripts': len(external_scripts)
        }
        
        print(f"   📊 Script Summary:")
        print(f"      • Total scripts: {result['total_scripts']}")
        print(f"      • With src: {result['src_scripts']}")
        print(f"      • Inline: {result['inline_scripts']}")
        print(f"      • External src: {result['external_scripts']}")
        
        return result
    
    def count_external_non_script_resources(self, soup, base_url):
        """นับ external resources ที่ไม่ใช่ script"""
        base_domain = urlparse(base_url).netloc
        external_resources = []
        
        # รายการ tags และ attributes ที่ต้องตรวจสอบ (ไม่รวม script)
        resource_tags = [
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
        
        print(f"\n🌐 External Non-Script Resources Analysis:")
        
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
                        print(f"   EXTERNAL {tag_name.upper()}: {src}")
            
            if count > 0:
                external_by_type[tag_name] = count
        
        print(f"   📊 External Non-Script Resources by Type:")
        for tag_type, count in external_by_type.items():
            print(f"      • {tag_type}: {count}")
        
        result = {
            'total_external': len(external_resources),
            'by_type': external_by_type,
            'details': external_resources
        }
        
        print(f"   📊 Total external non-script resources: {result['total_external']}")
        
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
        folder_path = f'csv/{list_type}'
        os.makedirs(folder_path, exist_ok=True)
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        filepath = f'{folder_path}/{filename}'
        
        # บันทึกเป็น CSV
        df = pd.DataFrame([data])
        df.to_csv(filepath, index=False, encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
        
        print(f"💾 Saved to: {filepath}")
        
        # แสดงตัวอย่าง string
        print(f"\n📋 Data String Preview (first 500 chars):")
        preview = data['data_string'][:500] + "..." if len(data['data_string']) > 500 else data['data_string']
        print(preview)
        
        # ตรวจสอบความสมบูรณ์ของข้อมูล
        lines = data['data_string'].split('\n')
        print(f"\n📊 Data completeness check:")
        print(f"   • Total lines in output: {len(lines)}")
        print(f"   • String length: {len(data['data_string'])} characters")
        
        # แสดงบรรทัดสุดท้ายเพื่อตรวจสอบ
        if lines:
            print(f"   • Last line: {lines[-1]}")

# การใช้งาน
if __name__ == "__main__":
    analyzer = HTMLAnalyzerComplete()
    
    # วิเคราะห์ Shopee
    print("🟢 Analyzing Shopee...")
    result = analyzer.analyze_single_url("https://playnows.co/register", "playnows", "test")
    
    if result:
        print("\n✅ Analysis completed!")
        print(f"📁 File saved")
    else:
        print("\n❌ Analysis failed!")