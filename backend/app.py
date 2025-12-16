# app.py - Flask REST API for Phishing Detection (Enhanced with Percentages)
# Run: python app.py  (default: http://localhost:5001)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import re
import time
import os
from datetime import datetime, date, timezone
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import whois
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import tldextract

app = Flask(__name__)
CORS(app)

# ============== CONFIG ==============
MODEL_PATH = os.getenv("MODEL_PATH", "./phishing_llm_newwy_3")
MAX_HTML_SIZE = None
WHOIS_TIMEOUT = 5
WHOIS_CACHE = {}
CACHE_TTL = 1800
REFERENCE_DATE = date(2025, 1, 1)

DISABLE_REPUTATION_GUARD = os.getenv("DISABLE_REPUTATION_GUARD", "0") == "1"

DOMAIN_WHITELIST = {
    "google.com", "gmail.com", "gstatic.com", "googleusercontent.com", "googleapis.com",
    "microsoft.com", "live.com", "outlook.com", "office.com", "office365.com", "msn.com",
    "apple.com", "icloud.com", "phishtank.com",
    "openai.com", "chatgpt.com", "oaiusercontent.com", "openaiapi-static.com",
    "anthropic.com", "claude.ai", "claudeusercontent.com",
    "facebook.com", "fbcdn.net", "twitter.com", "x.com", "twimg.com", "linkedin.com", "instagram.com",
    "cloudflare.com", "bootstrapcdn.com", "cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com",
    "akamaihd.net", "akamaized.net",
    "flaticon.com", "quillbot.com", "deepl.com", "grammarly.com",
}

SAFE_CDN_DOMAINS = {
    "gstatic.com", "googleusercontent.com", "googleapis.com",
    "cloudflare.com", "bootstrapcdn.com", "cdn.jsdelivr.net",
    "unpkg.com", "cdnjs.cloudflare.com", "akamaihd.net", "akamaized.net",
    "microsoft.com", "office.net", "office365.com", "msn.com",
    "fbcdn.net", "facebook.com", "twimg.com", "twitter.com", "x.com",
    "openai.com", "chatgpt.com", "oaiusercontent.com", "openaiapi-static.com",
    "anthropic.com", "claude.ai", "claudeusercontent.com",
}

# ============== LOAD MODEL ==============
print("Loading model...")
device = torch.device("cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()
print(f"Model loaded on {device}")

# ============== HTTP CLIENT ==============
def create_session():
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

http_session = create_session()

# ============== DOMAIN UTILS ==============
def get_registrable_domain(host: str) -> str:
    if not host:
        return ""
    ext = tldextract.extract(host)
    if not ext.domain or not ext.suffix:
        return host
    return f"{ext.domain}.{ext.suffix}"

def same_org(host_a: str, host_b: str) -> bool:
    if not host_a or not host_b:
        return False
    a = get_registrable_domain(host_a)
    b = get_registrable_domain(host_b)
    return a and b and (a == b)

def is_safe_cdn(host: str) -> bool:
    if not host:
        return False
    reg = get_registrable_domain(host)
    return any(reg == d or reg.endswith("." + d) for d in SAFE_CDN_DOMAINS)

# ============== HELPER FUNCTIONS ==============
def clean_value(v):
    if v is None:
        return ""
    s = str(v).replace('"', "").replace("'", "").replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"\s+", " ", s)

def extract_domain_from_url(url):
    if not url or url == "unknown_url":
        return "unknown_domain"
    try:
        host = urlparse(url).netloc
        host = host.replace("www.", "") if host else ""
        registrable = get_registrable_domain(host)
        return registrable or (host if host else "unknown_domain")
    except:
        return "unknown_domain"

def try_parse_date(s):
    s = (s or "").strip()
    fmts = [
        "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d", "%d-%b-%Y",
        "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ"
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None

def normalize_creation_dates(raw):
    def to_naive_dt(x):
        if x is None:
            return None
        if isinstance(x, str):
            dt = try_parse_date(x)
            return dt
        if isinstance(x, datetime):
            if x.tzinfo is not None:
                x = x.astimezone(timezone.utc).replace(tzinfo=None)
            return x
        return None

    if isinstance(raw, list):
        out = []
        for v in raw:
            nv = to_naive_dt(v)
            if nv:
                out.append(nv)
        return out
    else:
        nv = to_naive_dt(raw)
        return [nv] if nv else []

def get_domain_age_years_label(domain):
    try:
        if not domain or domain == "unknown_domain" or "." not in domain:
            return "suspect"

        host = get_registrable_domain(domain.lower().split("/")[0])

        if host in WHOIS_CACHE:
            age_label, ts = WHOIS_CACHE[host]
            if time.time() - ts < CACHE_TTL:
                return age_label

        print(f"WHOIS lookup: {host}")
        info = whois.whois(host)

        creation_raw = getattr(info, "creation_date", None)
        norm_list = normalize_creation_dates(creation_raw)

        if not norm_list:
            return "suspect"

        creation_dt = min(norm_list)
        creation_date_only = creation_dt.date()

        age_days = (REFERENCE_DATE - creation_date_only).days
        age_years = max(0.0, round(age_days / 365.25, 2))
        result = f"{age_years:.2f} years" if age_years > 0 else "suspect"

        WHOIS_CACHE[host] = (result, time.time())
        return result

    except Exception as e:
        print(f"WHOIS failed for {domain}: {e}")
        return "suspect"

def fetch_html_from_url(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = http_session.get(url, headers=headers, timeout=timeout,
                                    allow_redirects=True, verify=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type.lower():
            return None, f"Not HTML content: {content_type}"

        return response.text, None

    except requests.exceptions.Timeout:
        return None, "Request timeout"
    except requests.exceptions.SSLError:
        return None, "SSL certificate error"
    except requests.exceptions.ConnectionError:
        return None, "Connection failed"
    except requests.exceptions.HTTPError as e:
        return None, f"HTTP error: {e}"
    except Exception as e:
        return None, f"Error fetching URL: {str(e)}"

# ============== FEATURE EXTRACTION ==============
def extract_title(soup):
    t = soup.find("title")
    return t.get_text(strip=True) if t else ""

def analyze_icon_type(soup, base_url):
    sels = [
        'link[rel*="icon"]', 'link[rel="shortcut icon"]',
        'link[rel="apple-touch-icon"]', 'link[rel="favicon"]'
    ]
    try:
        base_host = urlparse(base_url).netloc if base_url else ""
    except:
        base_host = ""

    internal = external = 0
    for sel in sels:
        for link in soup.select(sel):
            href = link.get("href")
            if not href:
                continue
            try:
                abs_url = urljoin(base_url if base_url else "http://local/", href)
                dom = urlparse(abs_url).netloc
                if not dom or same_org(dom, base_host) or is_safe_cdn(dom):
                    internal += 1
                else:
                    external += 1
            except:
                continue

    if internal > 0 and external == 0:
        return "internal"
    if external > 0 and internal == 0:
        return "external"
    if internal > 0 and external > 0:
        return "mixed"
    return "none"

def count_scripts(soup, base_url):
    all_scripts = soup.find_all("script")
    try:
        base_host = urlparse(base_url).netloc if base_url else ""
    except:
        base_host = ""

    src_scripts = inline_scripts = external_scripts = 0
    for sc in all_scripts:
        src = sc.get("src")
        if src:
            src_scripts += 1
            try:
                if src.startswith("//"):
                    dom = urlparse("http:" + src).netloc
                elif src.startswith("http"):
                    dom = urlparse(src).netloc
                else:
                    dom = base_host

                if dom and (not same_org(dom, base_host)) and (not is_safe_cdn(dom)):
                    external_scripts += 1
            except:
                pass
        else:
            inline_scripts += 1

    return {
        "total": len(all_scripts),
        "src": src_scripts,
        "inline": inline_scripts,
        "external": external_scripts
    }

def count_external_resources(soup, base_url):
    try:
        base_host = urlparse(base_url).netloc if base_url else ""
    except:
        base_host = ""

    tags = [
        ("script", "src"), ("link", "href"), ("img", "src"),
        ("iframe", "src"), ("embed", "src"), ("object", "data"),
        ("source", "src"), ("video", "src"), ("audio", "src")
    ]

    total_external = 0
    for tag, attr in tags:
        for el in soup.find_all(tag):
            src = el.get(attr)
            if not src:
                continue
            try:
                if src.startswith("//"):
                    dom = urlparse("http:" + src).netloc
                elif src.startswith("http"):
                    dom = urlparse(src).netloc
                else:
                    dom = base_host

                if dom and (not same_org(dom, base_host)) and (not is_safe_cdn(dom)):
                    total_external += 1
            except:
                continue

    return total_external

def count_redirects(soup, base_url):
    redirect_count = 0
    meta_refreshes = soup.find_all("meta", {"http-equiv": "refresh"})
    redirect_count += len(meta_refreshes)

    js_patterns = [
        r"window\.location\s*=", r"window\.location\.href\s*=",
        r"window\.location\.replace\s*\(", r"location\.href\s*=",
        r"location\.replace\s*\("
    ]
    for script in soup.find_all("script"):
        if script.string:
            for pattern in js_patterns:
                redirect_count += len(re.findall(pattern, script.string, flags=re.I))

    try:
        base_host = urlparse(base_url).netloc if base_url else ""
        for form in soup.find_all("form"):
            action = form.get("action")
            if action and action.startswith("http"):
                form_domain = urlparse(action).netloc
                if form_domain and (not same_org(form_domain, base_host)) and (not is_safe_cdn(form_domain)):
                    redirect_count += 1
    except:
        pass

    return min(redirect_count, 5)

def count_popups(soup):
    patterns = [
        r"window\.open\s*\(", r"alert\s*\(", r"confirm\s*\(",
        r"\.modal\s*\(", r"showModal", r"openPopup"
    ]
    count = 0
    for sc in soup.find_all("script"):
        if not sc.string:
            continue
        for pat in patterns:
            count += len(re.findall(pat, sc.string, flags=re.I))
    return min(count, 5)

def build_data_string(url, html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    lines = []

    lines.append(f"URL: {url}")
    domain = extract_domain_from_url(url)
    lines.append(f"Domain: {domain}")
    lines.append(f"Domain_age: {get_domain_age_years_label(domain)}")
    lines.append(f"Title: {extract_title(soup)}")

    for m in soup.find_all("meta"):
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

        if key:
            lines.append(f"{key}: {clean_value(m.get('content', ''))}")

    icon_type = analyze_icon_type(soup, url)
    lines.append(f"Icon_type: {icon_type}")

    sc = count_scripts(soup, url)
    ext_total = count_external_resources(soup, url)
    redirect_count = count_redirects(soup, url)
    popup_count = count_popups(soup)

    lines.append(f"Number_of_redirect: {redirect_count}")
    lines.append(f"Number_of_popup: {popup_count}")
    lines.append(f"Number_of_script: {sc['total']}")
    lines.append(f"Number_of_src_script: {sc['src']}")
    lines.append(f"Number_of_inline_script: {sc['inline']}")
    lines.append(f"Number_of_external_script: {sc['external']}")
    lines.append(f"Number_of_external_resources: {ext_total}")

    return "\n".join(lines)

def clean_text_for_model(text):
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

@torch.no_grad()
def predict_phishing(data_string):
    text = clean_text_for_model(data_string)
    enc = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    outputs = model(**enc)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred = int(probs.argmax())
    phishing_prob = float(probs[1])
    legit_prob = float(probs[0])
    return pred, phishing_prob, legit_prob

# ============== REPUTATION GUARD ==============
def parse_age_years(label: str) -> float:
    try:
        if not label or "suspect" in label.lower():
            return 0.0
        m = re.search(r"([0-9]+(\.[0-9]+)?)\s*years", label.lower())
        if m:
            return float(m.group(1))
    except:
        pass
    return 0.0

def reputation_override(domain: str, phishing_prob: float, legit_prob: float, features: dict):
    if DISABLE_REPUTATION_GUARD:
        return None

    reg = get_registrable_domain(domain)
    if reg in DOMAIN_WHITELIST:
        return 0

    age_years = parse_age_years(features.get("domain_age", "")) if "domain_age" in features else 0.0
    redir = int(features.get("redirect_count", 0))
    pop = int(features.get("popup_count", 0))
    ext_scripts = int(features.get("external_script_count", 0))
    ext_res = int(features.get("external_resources", 0))
    mild_signals = (redir <= 1) and (pop <= 2) and (ext_scripts <= 10) and (ext_res <= 20)

    margin = phishing_prob - legit_prob
    if age_years >= 5.0 and mild_signals:
        if margin < 0.20:
            return 0

    return None

# ============== API ENDPOINTS ==============
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model": "loaded", "device": str(device)})

@app.route('/api/predict/simple', methods=['POST'])
def predict_simple():
    """
    Simple prediction endpoint with percentages
    Request JSON: {"html": "...", "url": "https://example.com"}
    Response JSON: {
        "prediction": 0|1,
        "phishing_percent": 45.23,
        "legitimate_percent": 54.77
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        url = (data.get('url') or '').strip()
        html_content = (data.get('html') or '').strip()

        if not url:
            return jsonify({"error": "URL is required"}), 400

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if not html_content:
            print(f"Fetching HTML from: {url}")
            html_content, error = fetch_html_from_url(url)
            if error:
                return jsonify({"error": f"Failed to fetch URL: {error}"}), 400
        else:
            print(f"Using provided HTML for: {url}")

        data_string = build_data_string(url, html_content)
        pred, ph, lg = predict_phishing(data_string)

        soup = BeautifulSoup(html_content, "html.parser")
        domain = extract_domain_from_url(url)
        domain_age = get_domain_age_years_label(domain)
        sc = count_scripts(soup, url)
        features = {
            "domain_age": domain_age,
            "redirect_count": count_redirects(soup, url),
            "popup_count": count_popups(soup),
            "external_script_count": sc['external'],
            "external_resources": count_external_resources(soup, url),
        }

        override = reputation_override(domain, ph, lg, features)
        if override is not None:
            pred = override

        # Convert to percentages
        phishing_percent = round(ph * 100, 2)
        legitimate_percent = round(lg * 100, 2)

        print(f"Prediction: {pred} ({'Phishing' if pred == 1 else 'Legitimate'})")
        print(f"Phishing: {phishing_percent}% | Legitimate: {legitimate_percent}%")
        
        return jsonify({
            "prediction": pred,
            "phishing_percent": phishing_percent,
            "legitimate_percent": legitimate_percent
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Full prediction endpoint with detailed information and percentages
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        url = (data.get('url') or '').strip()
        html_content = (data.get('html') or '').strip()

        if not url:
            return jsonify({"error": "URL is required"}), 400

        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        if not html_content:
            print(f"Fetching HTML from: {url}")
            html_content, error = fetch_html_from_url(url)
            if error:
                return jsonify({"error": f"Failed to fetch URL: {error}", "url": url}), 400
        else:
            print(f"Using provided HTML for: {url}")

        print("🔨 Building data string...")
        data_string = build_data_string(url, html_content)

        soup = BeautifulSoup(html_content, "html.parser")
        domain = extract_domain_from_url(url)
        domain_age = get_domain_age_years_label(domain)
        sc = count_scripts(soup, url)

        features = {
            "title": extract_title(soup),
            "icon_type": analyze_icon_type(soup, url),
            "redirect_count": count_redirects(soup, url),
            "popup_count": count_popups(soup),
            "script_count": sc['total'],
            "external_script_count": sc['external'],
            "external_resources": count_external_resources(soup, url),
            "domain_age": domain_age,
        }

        print("Predicting...")
        pred, phishing_prob, legit_prob = predict_phishing(data_string)

        override = reputation_override(domain, phishing_prob, legit_prob, features)
        if override is not None:
            pred = override

        # Convert to percentages
        phishing_percent = round(phishing_prob * 100, 2)
        legitimate_percent = round(legit_prob * 100, 2)

        max_prob = max(phishing_prob, legit_prob)
        confidence = "high" if max_prob >= 0.85 else "medium" if max_prob >= 0.65 else "low"

        response = {
            "prediction": pred,
            "label": "Phishing" if pred == 1 else "Legitimate",
            "phishing_probability": round(phishing_prob, 4),
            "legitimate_probability": round(legit_prob, 4),
            "phishing_percent": phishing_percent,
            "legitimate_percent": legitimate_percent,
            "confidence": confidence,
            "url": url,
            "domain": domain,
            "domain_age": domain_age,
            "features": features,
            "data_string_preview": (data_string[:200] + "...") if len(data_string) > 200 else data_string
        }

        print(f"Result: {response['label']} ({max_prob:.2%} confidence)")
        print(f"Percentages - Phishing: {phishing_percent}% | Legitimate: {legitimate_percent}%")
        return jsonify(response)

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint with percentages
    Request JSON: {"urls": ["https://example1.com", "https://example2.com"]}
    """
    try:
        data = request.get_json()
        urls = data.get('urls', [])

        if not urls or not isinstance(urls, list):
            return jsonify({"error": "urls must be a non-empty list"}), 400
        if len(urls) > 10:
            return jsonify({"error": "Maximum 10 URLs per request"}), 400

        results = []
        for url in urls:
            try:
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url

                html_content, error = fetch_html_from_url(url)
                if error:
                    results.append({"url": url, "error": error})
                    continue

                data_string = build_data_string(url, html_content)
                pred, ph, lg = predict_phishing(data_string)

                soup = BeautifulSoup(html_content, "html.parser")
                domain = extract_domain_from_url(url)
                domain_age = get_domain_age_years_label(domain)
                sc = count_scripts(soup, url)
                features = {
                    "domain_age": domain_age,
                    "redirect_count": count_redirects(soup, url),
                    "popup_count": count_popups(soup),
                    "external_script_count": sc['external'],
                    "external_resources": count_external_resources(soup, url),
                }
                override = reputation_override(domain, ph, lg, features)
                if override is not None:
                    pred = override

                # Convert to percentages
                phishing_percent = round(ph * 100, 2)
                legitimate_percent = round(lg * 100, 2)

                results.append({
                    "url": url,
                    "prediction": pred,
                    "label": "Phishing" if pred == 1 else "Legitimate",
                    "phishing_probability": round(ph, 4),
                    "legitimate_probability": round(lg, 4),
                    "phishing_percent": phishing_percent,
                    "legitimate_percent": legitimate_percent
                })
            except Exception as e:
                results.append({"url": url, "error": str(e)})

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ============== RUN ==============
if __name__ == '__main__':
    PORT = int(os.getenv('PORT', 5001))
    print("\n" + "="*60)
    print("Phishing Detection API Server (with Percentages)")
    print("="*60)
    print(f"Simple: http://localhost:{PORT}/api/predict/simple")
    print(f"Full:   http://localhost:{PORT}/api/predict")
    print(f"Health: http://localhost:{PORT}/health")
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {device}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)
