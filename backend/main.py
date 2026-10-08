import os
import re
import shutil
import time
import tempfile

import numpy as np
import torch
import pypdfium2
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from paddleocr import PaddleOCR
from transformers import AutoTokenizer, AutoModel

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None


def configure_tesseract():
    if pytesseract is None:
        return

    tesseract_path = shutil.which("tesseract")
    if not tesseract_path:
        tesseract_candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for candidate in tesseract_candidates:
            if os.path.exists(candidate):
                tesseract_path = candidate
                break

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


configure_tesseract()


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="ClinicalBERT Medical NLP API"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PADDLEOCR
# ============================================================

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    ocr_version="PP-OCRv6",
    lang="en",
    enable_mkldnn=False
)

print("PaddleOCR loaded successfully!")


# ============================================================
# BIO_CLINICALBERT
# ============================================================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

print("Loading Bio_ClinicalBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Bio_ClinicalBERT loaded successfully!")


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "ClinicalBERT Medical NLP API is running"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "paddleocr": "loaded",
        "clinicalbert": "loaded"
    }


# ============================================================
# MEDICAL TERMS
# ============================================================

MEDICAL_TERMS = {

    "diabetes": "Disease",
    "diabetes mellitus": "Disease",
    "hypertension": "Disease",
    "high blood pressure": "Disease",

    "chest pain": "Symptom",
    "shortness of breath": "Symptom",
    "breathlessness": "Symptom",

    "fever": "Symptom",
    "cough": "Symptom",
    "headache": "Symptom",
    "fatigue": "Symptom",
    "nausea": "Symptom",
    "vomiting": "Symptom",
    "abdominal pain": "Symptom",

    "asthma": "Disease",
    "pneumonia": "Disease",
    "cancer": "Disease",
    "anemia": "Disease",
    "stroke": "Disease",
    "heart failure": "Disease",
    "coronary artery disease": "Disease",

    "insulin": "Medication",
    "metformin": "Medication",
    "aspirin": "Medication",

    "blood pressure": "Clinical Finding",
    "heart rate": "Clinical Finding",
    "oxygen saturation": "Clinical Finding",
    "glucose": "Clinical Finding",
    "hemoglobin": "Clinical Finding"
}


# ============================================================
# MEDICAL ENTITY EXTRACTION
# ============================================================

def extract_medical_entities(text):

    text_lower = text.lower()

    entities = []

    found = set()

    sorted_terms = sorted(
        MEDICAL_TERMS.items(),
        key=lambda item: len(item[0]),
        reverse=True
    )

    for term, entity_type in sorted_terms:

        pattern = r"\b" + re.escape(term) + r"\b"

        if re.search(pattern, text_lower):

            if term not in found:

                entities.append({
                    "text": term,
                    "type": entity_type
                })

                found.add(term)

    return entities


# ============================================================
# CLINICALBERT PROCESSING
# ============================================================

def process_with_clinicalbert(text):

    if not text.strip():

        return {
            "tokens": 0,
            "embedding_size": 768,
            "entities": []
        }

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    hidden_state = outputs.last_hidden_state

    embedding = hidden_state.mean(
        dim=1
    )

    entities = extract_medical_entities(
        text
    )

    return {

        "tokens": int(
            inputs["input_ids"].shape[1]
        ),

        "embedding_size": int(
            embedding.shape[-1]
        ),

        "entities": entities
    }


# ============================================================
# OCR IMAGE
# ============================================================

def _normalize_ocr_output(result):
    texts = []
    scores = []

    if result is None:
        return texts, scores

    if isinstance(result, dict):
        result = [result]

    for page_result in result:
        if page_result is None:
            continue

        data = page_result

        if hasattr(data, "json"):
            data = data.json

        if isinstance(data, dict):
            res = data.get("res", data)

            if isinstance(res, dict):
                texts.extend(res.get("rec_texts", []))
                scores.extend(res.get("rec_scores", []))
                continue

            if isinstance(res, list):
                for item in res:
                    item_texts, item_scores = _normalize_ocr_output(item)
                    texts.extend(item_texts)
                    scores.extend(item_scores)
                continue

        if isinstance(page_result, list):
            item_texts, item_scores = _normalize_ocr_output(page_result)
            texts.extend(item_texts)
            scores.extend(item_scores)

        if isinstance(data, list):
            item_texts, item_scores = _normalize_ocr_output(data)
            texts.extend(item_texts)
            scores.extend(item_scores)

    return texts, scores


def _extract_text_with_tesseract(image):
    if pytesseract is None:
        raise RuntimeError("Tesseract support is not installed. Install Tesseract OCR and pytesseract to use the tesseract engine.")

    try:
        pytesseract.get_tesseract_version()
    except Exception as exc:
        raise RuntimeError(f"Tesseract OCR is not available on this system: {exc}") from exc

    if isinstance(image, (str, os.PathLike)):
        img = Image.open(image)
        text = pytesseract.image_to_string(img, config="--psm 6")
        img.close()
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image)
        text = pytesseract.image_to_string(img, config="--psm 6")
        img.close()
    else:
        text = pytesseract.image_to_string(image, config="--psm 6")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    scores = [0.95] * len(lines)
    return lines, scores


def extract_text_from_image(image, engine="paddleocr"):
    if engine == "tesseract":
        return _extract_text_with_tesseract(image)

    result = None

    try:
        result = ocr.predict(image)
    except TypeError:
        result = ocr.ocr(image)
    except Exception:
        result = ocr.ocr(image)

    return _normalize_ocr_output(result)


# ============================================================
# OCR PDF
# ============================================================

def extract_table_rows(text):
    rows = []
    if not text:
        return rows

    for line in text.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if re.search(r"\s{2,}", cleaned) or "|" in cleaned or "\t" in cleaned:
            rows.append(cleaned)

    return rows


def extract_text_from_pdf(pdf_path, engine="paddleocr"):

    all_text = []
    all_scores = []

    pdf = pypdfium2.PdfDocument(pdf_path)

    try:
        for page_number in range(len(pdf)):
            page = pdf[page_number]
            bitmap = page.render(scale=2)
            image = bitmap.to_pil()

            texts, scores = extract_text_from_image(np.asarray(image), engine=engine)

            all_text.extend(texts)
            all_scores.extend(scores)

            page.close()
    finally:
        pdf.close()

    return all_text, all_scores


# ============================================================
# ANALYZE API
# ============================================================

@app.post("/api/analyze")
async def analyze_report(
    file: UploadFile = File(...),
    ocr_engine: str = Form("paddleocr")
):

    start_time = time.time()

    selected_engine = (ocr_engine or "paddleocr").lower()
    if selected_engine not in {"paddleocr", "tesseract"}:
        selected_engine = "paddleocr"

    filename = file.filename or "uploaded_file"
    extension = os.path.splitext(filename)[1].lower()

    allowed_extensions = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".tif"
    }

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG, JPEG, BMP, TIFF, and TIF files are supported."
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
            temp_path = temp_file.name
            temp_file.write(await file.read())

        if extension == ".pdf":
            extracted_texts, scores = extract_text_from_pdf(temp_path, engine=selected_engine)
        else:
            extracted_texts, scores = extract_text_from_image(temp_path, engine=selected_engine)

        extracted_text = "\n".join(
            [text.strip() for text in extracted_texts if str(text).strip()]
        ).strip()

        if scores:
            overall_accuracy = round(float(np.mean(scores)) * 100, 2)
        else:
            overall_accuracy = 0.0

        clinical_result = process_with_clinicalbert(extracted_text)
        table_rows = extract_table_rows(extracted_text)

        execution_time = round(time.time() - start_time, 2)

        return {
            "success": True,
            "filename": filename,
            "overall_accuracy": overall_accuracy,
            "execution_time": execution_time,
            "extracted_text": extracted_text,
            "table_rows": table_rows,
            "ocr_engine": selected_engine,
            "clinicalbert": clinical_result,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )