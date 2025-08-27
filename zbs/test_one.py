from transformers import BertTokenizer, BertForSequenceClassification
import torch
import pandas as pd
import os

# === CONFIG ===
model_path = "./phishing_llm_newwy"
device = torch.device("cpu")
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path).to(device).eval()

# === FUNCTION ===
def predict_with_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        phishing_score = probs[0][1].item()
        legit_score = probs[0][0].item()
    return pred, phishing_score, legit_score

# === MANUAL FILE LIST FROM LOG ===
test_files = [
    # 🟢 White (label 0)
    {"path": "csv_train/white/row_344_www.vestiairecollective.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_309_www.olacabs.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_288_bohdan-books.com_foreign-rights__srsltid=AfmBOoo0AHa-JgpbohQygIeESem-o1CvgWT46yJ5xnTg1tjzBiNEl8FT_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_73_www.tedsmontanagrill.com__gad_source=1&gad_campaignid=22799456150&gclid=Cj0KCQjwyvfDBhDYARIsAItzbZFc_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_119_www.temu.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_325_www.idealista.com_en__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_108_global.alipay.com_platform_site_ihome_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_331_wellhub.com_en-us__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_41_deliveroo.co.uk__srsltid=AfmBOop31pV5I-74puyfh1xXhy4aty1MfULkpITIe88Z5nDBT4LTZfJ8_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_231_www.transartists.org_en_organisation_mann-ivanov-and-ferber_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_326_www.uniplaces.com__srsltid=AfmBOoq8qlcaS-NlDkkqpbh8uDaJGnSLVt3A5T-0wQlnulVRNhBoo9Zb_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_29_www.mcdonalds.co.th__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_272_starylev.com.ua_about-us_srsltid=AfmBOooKl4ewpSh0cxnKn0BsyqiSgBXzTm2rgCCiRfwSMWrkh2vSFx4o_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_157_www.belpost.by_en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_280_pocketbook.ch_en-ch_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_240_play.google.com_store_apps_dev_id=8427637182851887164&hl=en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_43_justeattakeaway.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_333_www.domestika.org_en_courses_popular_utm_source=google&utm_medium=cpc&utm_campaign=01_OTHERS_MIX_NA__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_382_banco.bradesco_html_classic_index.shtm_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_88_sezzledealsnow.com__source=_2Bsezzle&gad_source=1&gad_campaignid=21705159098&gclid=Cj0KCQjwyvfDBhDYA_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_274_chytomo.com_en_the-ukrainian-old-lion-publishing-house-launches-print-on-demand-project__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_4_www.thaiairways.com_en-th__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_67_n26.com_en-eu_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_334_www.vendia.com_use-cases_data-monetization__utm_source=google&utm_medium=paid_search&utm_campaign=gl_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_233_www.mann-ivanov-ferber.com__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_145_cdek-th.com_en_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_178_5ka.ru__utm_referrer=https_3a_2f_2fwww.google.com_2f_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_192_www.exlibrus.de__q=en_shop_book_list_publisher_Vagrius_analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_210_rosman.ru_en__analyzed.csv", "label": 0},
    {"path": "csv_train/white/row_256_www.ebay.com__analyzed.csv", "label": 0},

    # 🔴 Black (label 1)
    {"path": "csv_train/black/row_87_drive.usercontent.google.com_download_id=1nWsqALU5AljotDVZ4LzE3992QiaSSLDZ&export=download_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_246_flowing_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_110_blinq_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_147_sso-itrustcapetal_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_62_aff.redkong89.cc__ref=418zqoc3g7_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_240_bafybeielj_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_319_excduswale_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_312_wirett.weebly_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_107_webproxy_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_337_tech.aton40_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_161_kquinelogin_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_10_simplyconnect.polygonuk.com__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_27_www.quickexchange.net__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_221_one_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_283_sidneymccray_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_88_soanywbha.com__analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_105_docs.google_sentform_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_299_fluffy-paprenjak-2518cc_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_385_s.team-yh_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_255_liambiggs201_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_103_att-mail-109008_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_141_bpccsdf_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_288_scribehoe_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_321_eu.jotform_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_35_www.gopumpkin.fun_airdrop_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_190_yuh_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_214_cpcontacts_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_228_westfield_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_250_get_analyzed.csv", "label": 1},
    {"path": "csv_train/black/row_265_wk-drilling_analyzed.csv", "label": 1},
]

# === RUN TESTS ===
correct = 0
print("🔍 Predicting manually listed test files...\n")

for f in test_files:
    path = f["path"]
    label = f["label"]
    name = os.path.basename(path)

    try:
        df = pd.read_csv(path)
        text = " ".join(df[df.columns[0]].astype(str).tolist()).strip()

        if not text:
            print(f"⚠️ Empty file: {name}\n")
            continue

        pred, phishing_score, legit_score = predict_with_score(text)

        print(f"🧪 Test: {name}")
        print(f"   ➤ Prediction: {'Phishing ⚠️' if pred == 1 else 'Legitimate ✅'}")
        print(f"   ➤ Confidence - Phishing: {phishing_score:.2%}, Legitimate: {legit_score:.2%}")
        print(f"   ➤ Ground Truth: {'Phishing (1)' if label == 1 else 'Legitimate (0)'}\n")

        if pred == label:
            correct += 1

    except Exception as e:
        print(f"❌ Error reading {name}: {e}")

# === SUMMARY ===
total = len(test_files)
print("===================================================")
print(f"✅ Total tested: {total}")
print(f"🎯 Accuracy: {correct / total:.2%}")
print("===================================================")
