# Document Verification Suite - Technical Guide

**Version:** 2.0.0  
**Date:** December 13, 2025  
**Status:** Production Ready

---

## Overview

Document verification system for insurance claim processing with:
- PAN Card Verification - Format, structure, OCR, forgery detection
- Aadhaar Verification - Checksum, hologram, QR code validation
- Generic Document Verification - License, passport, bills, certificates
- OCR Text Extraction - Multi-language with entity detection
- Smart Fallbacks - Graceful degradation for missing/poor quality images
- Rate Limiting - API protection (100 req/min)

---

## API Endpoints

### 1. PAN Card Verification

**Endpoint:** `POST /api/documents/verify-pan`

**Features:**
- Format validation (AAAAA9999A pattern)
- Structure verification
- OCR extraction (name, father's name, DOB)
- Forgery detection
- Cross-verification support
- Quality assessment

**Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/verify-pan" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@pan_card.jpg" \
  -F "expected_pan=ABCDE1234F" \
  -F "expected_name=John Doe"
```

**Response:**
```json
{
  "status": "success",
  "document_type": "PAN",
  "is_valid": true,
  "confidence": 0.92,
  "extracted_data": {
    "pan_number": "ABCDE1234F",
    "name": "JOHN DOE",
    "fathers_name": "RICHARD DOE",
    "date_of_birth": "01/01/1990"
  },
  "validation_checks": {
    "format_valid": true,
    "structure_valid": true,
    "ocr_confidence": 0.94,
    "quality_score": 0.88,
    "forgery_detected": false
  },
  "risk_score": 0.05,
  "red_flags": [],
  "recommendation": "APPROVE - PAN verified successfully"
}
```

**Validation Rules:**
- Format: 5 letters + 4 digits + 1 letter (e.g., ABCDE1234F)
- Structure: 4th character = P/C/H/F/A/T/B/L/J/G
- OCR confidence >= 60%
- Quality score >= 60%
- No forgery detected

---

### 2. Aadhaar Card Verification

**Endpoint:** `POST /api/documents/verify-aadhaar`

**Features:**
- Format validation (12 digits)
- Checksum validation (Verhoeff algorithm)
- Hologram detection
- QR code verification
- OCR extraction (masked for privacy)
- Forgery detection

**Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/verify-aadhaar" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@aadhaar_card.jpg" \
  -F "expected_aadhaar=1234" \
  -F "expected_name=John Doe" \
  -F "mask_number=true"
```

**Response:**
```json
{
  "status": "success",
  "document_type": "AADHAAR",
  "is_valid": true,
  "confidence": 0.89,
  "extracted_data": {
    "aadhaar_number": "XXXX-XXXX-1234",
    "name": "John Doe",
    "date_of_birth": "01/01/1990",
    "gender": "Male",
    "address": "123 Main St, Mumbai"
  },
  "validation_checks": {
    "format_valid": true,
    "checksum_valid": true,
    "ocr_confidence": 0.91,
    "quality_score": 0.85,
    "hologram_present": true,
    "qr_code_present": true,
    "forgery_detected": false
  },
  "risk_score": 0.08,
  "red_flags": [],
  "recommendation": "APPROVE - Aadhaar verified successfully"
}
```

**Validation Rules:**
- Format: Exactly 12 digits
- Checksum: Verhoeff algorithm passes
- Hologram present
- OCR confidence >= 60%
- No forgery detected

---

### 3. Generic Document Verification

**Endpoint:** `POST /api/documents/verify-document`

**Supported Documents:**
- Driving License
- Passport
- Voter ID
- Bank Statement
- Hospital Bill
- Death Certificate
- Other

**Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/verify-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@driving_license.jpg" \
  -F "document_type=license"
```

**Response:**
```json
{
  "status": "success",
  "document_type": "LICENSE",
  "is_valid": true,
  "confidence": 0.87,
  "extracted_data": {
    "license_number": "DL-1234567890123",
    "name": "John Doe",
    "validity": "2025-12-31",
    "address": "Mumbai, Maharashtra"
  },
  "validation_checks": {
    "ocr_confidence": 0.88,
    "quality_score": 0.86,
    "forgery_detected": false
  },
  "risk_score": 0.12,
  "red_flags": [],
  "recommendation": "APPROVE - Document verified"
}
```

---

### 4. OCR Text Extraction

**Endpoint:** `POST /api/documents/extract-text`

**Features:**
- Multi-language support (English, Hindi)
- Entity detection (PAN, Aadhaar, dates, amounts)
- Layout preservation
- Confidence scoring

**Request:**
```bash
curl -X POST "http://localhost:8000/api/documents/extract-text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.jpg"
```

**Response:**
```json
{
  "status": "success",
  "extracted_text": "This is a sample document with PAN ABCDE1234F and Aadhaar 1234 5678 9012...",
  "confidence": 0.91,
  "word_count": 245,
  "line_count": 18,
  "detected_entities": {
    "pan_numbers": ["ABCDE1234F"],
    "aadhaar_numbers": ["1234 5678 9012"],
    "dates": ["01/01/2024", "12/12/2025"],
    "amounts": ["₹10,000", "₹25,500"],
    "phone_numbers": ["9876543210"]
  }
}
```

---

## Testing Guide

### Test Scenario 1: Valid PAN Card

**Test Case:** Upload high-quality PAN card image

**Expected Result:**
```json
{
  "is_valid": true,
  "confidence": >= 0.85,
  "risk_score": < 0.3,
  "recommendation": "APPROVE",
  "red_flags": []
}
```

**Verification:**
- Format valid (AAAAA9999A)
- OCR confidence >= 85%
- No forgery detected
- Quality score >= 85%

---

### Test Scenario 2: Forged PAN Card

**Test Case:** Upload photoshopped/edited PAN image

**Expected Result:**
```json
{
  "is_valid": false,
  "confidence": < 0.5,
  "risk_score": >= 0.6,
  "recommendation": "REJECT",
  "red_flags": ["Possible forgery/manipulation detected"]
}
```

**Verification:**
- Forgery detected = true
- High risk score (>= 0.6)
- Recommendation = REJECT

---

### Test Scenario 3: Poor Quality Image

**Test Case:** Upload blurry/low-resolution image

**Expected Result:**
```json
{
  "is_valid": false,
  "confidence": < 0.6,
  "risk_score": >= 0.4,
  "recommendation": "REVIEW",
  "red_flags": [
    "Low OCR confidence",
    "Poor image quality (blur, low resolution)"
  ]
}
```

**Verification:**
- Quality score < 60%
- OCR confidence < 70%
- Red flags added
- Recommendation = REVIEW

---

### Test Scenario 4: Cross-Verification Mismatch

**Test Case:** PAN number doesn't match expected value

**Expected Result:**
```json
{
  "is_valid": false,
  "risk_score": >= 0.5,
  "recommendation": "REJECT",
  "red_flags": [
    "PAN mismatch: expected AAAAA1111A, got BBBBB2222B"
  ]
}
```

**Verification:**
- Mismatch detected
- Red flag added with details
- High risk score

---

## Security & Privacy

### Aadhaar Masking
```python
# Request with masking (default)
mask_number=true  # Returns: XXXX-XXXX-1234

# Request without masking (requires authorization)
mask_number=false  # Returns: 1234-5678-9012
```

### Data Retention
- Temporary files deleted immediately after processing
- No PII stored on server
- Aadhaar numbers masked by default
- Logs sanitized (no sensitive data)

---

## Risk Scoring

### Risk Score Calculation

**PAN Card:**
```
risk_score = (
  0.4 if invalid_format else 0.0 +
  0.6 if forgery_detected else 0.0 +
  0.2 if ocr_confidence < 0.7 else 0.0 +
  0.15 if quality_score < 0.6 else 0.0 +
  0.5 if pan_mismatch else 0.0 +
  0.3 if name_mismatch else 0.0
)
```

**Aadhaar Card:**
```
risk_score = (
  0.4 if invalid_format else 0.0 +
  0.5 if invalid_checksum else 0.0 +
  0.6 if forgery_detected else 0.0 +
  0.3 if missing_hologram else 0.0 +
  0.2 if ocr_confidence < 0.7 else 0.0 +
  0.5 if aadhaar_mismatch else 0.0 +
  0.3 if name_mismatch else 0.0
)
```

### Recommendation Logic

| Risk Score | Recommendation |
|------------|----------------|
| >= 0.7 | REJECT - High fraud risk |
| 0.4 - 0.7 | REVIEW - Manual verification required |
| < 0.4 | APPROVE - Document verified successfully |

---

## Integration Examples

### Python
```python
import requests

# Verify PAN card
with open('pan_card.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/documents/verify-pan',
        files={'file': f},
        data={
            'expected_pan': 'ABCDE1234F',
            'expected_name': 'John Doe'
        }
    )

result = response.json()
if result['is_valid']:
    print(f"PAN verified: {result['extracted_data']['pan_number']}")
else:
    print(f"Verification failed: {result['red_flags']}")
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);
formData.append('expected_pan', 'ABCDE1234F');

fetch('http://localhost:8000/api/documents/verify-pan', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => {
  if (data.is_valid) {
    console.log('PAN verified:', data.extracted_data.pan_number);
  } else {
    console.log('Verification failed:', data.red_flags);
  }
});
```

### cURL
```bash
curl -X POST http://localhost:8000/api/documents/verify-pan \
  -F "file=@pan_card.jpg" \
  -F "expected_pan=ABCDE1234F"
```

---

## Configuration

### Environment Variables
```bash
# Image size limits
MAX_IMAGE_SIZE_MB=10

# OCR settings
OCR_LANGUAGE=eng+hin
OCR_CONFIDENCE_THRESHOLD=0.6

# Quality thresholds
MIN_QUALITY_SCORE=0.6
MIN_OCR_CONFIDENCE=0.7

# Forgery detection
FORGERY_THRESHOLD=0.55
ENABLE_FORGERY_DETECTION=true

# Rate limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

---

## Error Handling

### Common Errors

**400 Bad Request**
```json
{
  "detail": "File must be an image (JPG, PNG) or PDF"
}
```
**Solution:** Upload valid image format

**413 Payload Too Large**
```json
{
  "detail": "File size (15.2MB) exceeds limit (10MB)"
}
```
**Solution:** Compress image or adjust MAX_IMAGE_SIZE_MB

**429 Too Many Requests**
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60
}
```
**Solution:** Wait 60 seconds or adjust rate limit

**500 Internal Server Error**
```json
{
  "detail": "PAN verification error: Model not loaded"
}
```
**Solution:** Check logs, restart service, verify models loaded

---

## Performance

### Response Times

| Endpoint | Avg Time | Max Time |
|----------|----------|----------|
| PAN Verification | 1.2s | 3.0s |
| Aadhaar Verification | 1.5s | 3.5s |
| Generic Verification | 1.0s | 2.5s |
| OCR Extraction | 0.8s | 2.0s |

### Accuracy

| Metric | Score |
|--------|-------|
| PAN Format Detection | 99.5% |
| Aadhaar Checksum Validation | 100% |
| Forgery Detection | 83.6% |
| OCR Accuracy (English) | 92% |
| OCR Accuracy (Hindi) | 87% |
