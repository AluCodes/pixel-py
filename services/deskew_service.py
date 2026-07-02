"""
Deskew Service Module
A FastAPI router module for image deskewing functionality.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional
import io
import os
import statistics
import time
import base64

import cv2
import numpy as np
import pytesseract

# Create router for this service
router = APIRouter(prefix="/deskew", tags=["deskew"])

# =================== CONFIG ===================
TEXT_MARGIN_THRESHOLD = 10
ASPECT_RATIO_RANGE = (0.3, 15.0)
MIN_AREA_RATIO = 0.01
MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_DIMENSION = 4096
# When two orientations have area within this ratio, prefer the one matching OSD (90° detection).
OSD_TIE_RATIO = 0.98
DEBUG = True


def check_tesseract() -> dict:
    """
    Verify Tesseract is installed and on PATH.
    Returns {"available": True, "version": "5.3.0"} or {"available": False, "error": "..."}.
    """
    try:
        version = pytesseract.get_tesseract_version()
        return {"available": True, "version": str(version)}
    except pytesseract.TesseractNotFoundError as e:
        return {"available": False, "error": "Tesseract is not installed or not in your PATH. Install it (e.g. brew install tesseract on macOS) and ensure the 'tesseract' binary is on PATH."}
    except Exception as e:
        return {"available": False, "error": str(e)}


# =================== MODELS ===================
class DeskewResponse(BaseModel):
    """Response model for deskew endpoint."""
    image: str = Field(..., description="Base64 encoded processed image")
    detected_angle: float = Field(..., description="Detected skew angle in degrees")
    corrected: bool = Field(..., description="Whether image was corrected")
    format: str = Field(..., description="Output image format")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


# =================== HELPER FUNCTIONS ===================
def debug_print(msg):
    if DEBUG:
        print(msg)

def order_points(pts: np.ndarray) -> np.ndarray:
    """Orders the four points of a contour in TL, TR, BR, BL order."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray):
    """Applies a perspective transform to flatten the image region."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, rect, maxWidth, maxHeight


def detect_text_margins(gray_img: np.ndarray, threshold: int = TEXT_MARGIN_THRESHOLD):
    """Estimates text boundaries by projecting pixel density."""
    h, w = gray_img.shape
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = 255 - binary

    vertical = np.sum(binary, axis=0)
    horizontal = np.sum(binary, axis=1)

    left_text = np.argmax(vertical > 0)
    right_text = w - np.argmax(vertical[::-1] > 0)
    top_text = np.argmax(horizontal > 0)
    bottom_text = h - np.argmax(horizontal[::-1] > 0)

    return {
        'left': max(0, threshold - left_text),
        'right': max(0, threshold - (w - right_text)),
        'top': max(0, threshold - top_text),
        'bottom': max(0, threshold - (h - bottom_text))
    }


def apply_directional_margin(rect: np.ndarray, margins: dict, width: int, height: int):
    """Expands the 4 corners of the detected document outward by the margin."""
    tl, tr, br, bl = rect

    if margins['left'] > 0:
        tl[0] -= margins['left']
        bl[0] -= margins['left']
    if margins['right'] > 0:
        tr[0] += margins['right']
        br[0] += margins['right']
    if margins['top'] > 0:
        tl[1] -= margins['top']
        tr[1] -= margins['top']
    if margins['bottom'] > 0:
        bl[1] += margins['bottom']
        br[1] += margins['bottom']

    points = np.array([tl, tr, br, bl])
    points[:, 0] = np.clip(points[:, 0], 0, width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, height - 1)
    return points


def text_detection_score(
    image: np.ndarray,
    max_side: int = 1600,
    min_side: int = 300,
    debug: bool | None = None,
) -> float:
    """
    Score how well text is detected in this orientation (0–100).
    Uses Tesseract word-level confidence; upright text yields higher scores.
    Image is resized for speed (max_side) but kept readable (min_side). Returns 0 if no words detected.

    Set DESKEW_DEBUG_TEXT=1 in the environment (or debug=True) to print diagnostics to stderr.
    """
    if image.size == 0:
        return 0.0

    h, w = image.shape[:2]
    
    debug_print(f"[text_detection_score] input shape={image.shape} dtype={image.dtype}")

    if len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV BGR -> RGB for Tesseract
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    longest = max(h, w)
    scale = 1.0
    if longest > max_side:
        scale = max_side / longest
    elif longest < min_side and longest > 0:
        scale = min_side / longest
    if scale != 1.0:
        new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    debug_print(f"[text_detection_score] after resize shape={image.shape}")

    # Pytesseract is more reliable with a PIL Image than a numpy array on many systems
    try:
        from PIL import Image as PILImage
        pil_image = PILImage.fromarray(image)
    except Exception:
        pil_image = image

    try:
        # Try PSM 6 first (single block of text), then 3 (fully auto), then 4 (single column)
        for psm in (6, 3, 4):
            config = f"--psm {psm}"
            data = pytesseract.image_to_data(pil_image, lang="eng", config=config, output_type=pytesseract.Output.DICT)

            confs = []
            for c in data.get("conf", []):
                try:
                    val = int(float(c)) if not isinstance(c, (int, float)) else int(c)
                    if val >= 0:
                        confs.append(val)
                except (ValueError, TypeError):
                    continue

            if confs:
                mean_conf = statistics.fmean(confs)
                median_conf = statistics.median(confs)
                modes = statistics.multimode(confs)
                mode_conf = statistics.fmean(modes) if len(modes) > 1 else float(modes[0])
                score = (mean_conf + median_conf + mode_conf) / 3.0

                debug_print(
                    f"[text_detection_score] confs n={len(confs)} mean={mean_conf:.2f} median={median_conf:.2f} mode={mode_conf:.2f} score={score:.2f}"
                )
                return min(100.0, max(0.0, score))

        # No PSM gave word confidences; try image_to_string as last resort
        text = pytesseract.image_to_string(pil_image, lang="eng", config="--psm 3")
        
        debug_print(f"[text_detection_score] no confs; fallback image_to_string len={len(text)} strip={repr(text.strip()[:200])}")
        if text and text.strip():
            return 10.0
        return 0.0
    except Exception as e:
        debug_print(f"[text_detection_score] exception: {e}")
        return 0.0


def get_osd_rotation(image: np.ndarray) -> Optional[int]:
    """
    Use Tesseract OSD to detect how much the image is rotated (0, 90, 180, or 270).
    Returns the rotation in degrees needed to make text upright, or None if OSD fails.
    """
    try:
        osd = pytesseract.image_to_osd(image, lang='osd')
        for line in osd.split('\n'):
            if 'Rotate:' in line:
                angle = int(line.split(':')[1].strip())
                debug_print(f"OSD detected rotation: {angle}°")
                return angle if angle in (0, 90, 180, 270) else None
        return 0
    except Exception:
        return None


def is_upright(image):
    """Use pytesseract to detect orientation of the image."""
    try:
        osd = pytesseract.image_to_osd(image, lang='osd')
        for line in osd.split('\n'):
            if 'Rotate:' in line:
                angle = int(line.split(':')[1].strip())
                debug_print(f"OSD detected rotation needed: {angle}°")
                # If OSD says rotate 180°, image is upside down
                return angle == 0, angle
        return True, 0
    except Exception as e:
        debug_print(f"Orientation detection failed: {e}")
        return True, 0


def rotate_img(img: np.ndarray, angle: int):
    """Rotate image by standard angles."""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img.copy()


def compute_best_skew_angle(image: np.ndarray):
    """Compute the best skew angle for fine-tuning."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scores = []
    for angle in np.arange(-40, 40.5, 0.5):
        M = cv2.getRotationMatrix2D((gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0)
        rotated = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        hist = np.sum(rotated, axis=1)
        score = np.sum((hist[1:] - hist[:-1]) ** 2)
        scores.append((score, angle))
    best_score, best_angle = max(scores)
    return best_angle


# =================== PREPROCESSING STAGES (standalone, enable/disable per stage) ===================


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Stage 1: CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances faint edges by improving local contrast.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def apply_gaussian_blur(
    gray: np.ndarray,
    kernel_size: tuple[int, int] = (5, 5),
    sigma_x: float = 0,
) -> np.ndarray:
    """
    Stage 2: Gaussian blur to reduce noise before edge detection.
    """
    kx, ky = kernel_size
    if kx % 2 == 0 or ky % 2 == 0:
        kx, ky = max(3, kx | 1), max(3, ky | 1)
    return cv2.GaussianBlur(gray, (kx, ky), sigma_x)


def apply_canny(
    gray: np.ndarray,
    low_threshold: int = 75,
    high_threshold: int = 200,
) -> np.ndarray:
    """
    Stage 3: Canny edge detection.
    """
    return cv2.Canny(gray, low_threshold, high_threshold)


def apply_morphological_gradient(
    edged: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
) -> np.ndarray:
    """
    Stage 4: Morphological gradient (optional refinement).
    Strengthens edge boundaries; use after Canny to refine the edge map.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(edged, cv2.MORPH_GRADIENT, kernel)


def detect_largest_quadrilateral(
    edged: np.ndarray,
    image_area: int,
    min_area_ratio: float = MIN_AREA_RATIO,
    aspect_ratio_range: tuple[float, float] = ASPECT_RATIO_RANGE,
    approx_epsilon_ratio: float = 0.02,
) -> Optional[np.ndarray]:
    """
    Stage 5: Contour detection — find the largest quadrilateral contour.
    Returns the 4-point contour (shape (4, 1, 2)) or None if none valid.
    """
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best: Optional[tuple[float, np.ndarray]] = None

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, approx_epsilon_ratio * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h
        aspect_ratio = h / float(w) if w else 0

        if area < min_area_ratio * image_area:
            continue
        if not (aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
            continue

        if best is None or area > best[0]:
            best = (float(area), approx)

    return best[1] if best is not None else None


def deskew_image(image: np.ndarray, angle: float):
    """Apply deskew rotation to the image."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def find_upright_orientation(image):
    """
    Try OSD on all 4 orientations of the image.
    Returns the angle needed to rotate to make it upright (0, 90, 180, 270).
    Returns 0 if detection fails.
    """
    best_confidence = -1
    best_angle = 0
    
    for test_angle in [0, 90, 180, 270]:
        test_img = rotate_img(image, test_angle)
        try:
            osd = pytesseract.image_to_osd(test_img, lang='osd')
            
            # Parse OSD output
            rotate_angle = 0
            confidence = 0
            for line in osd.split('\n'):
                if 'Rotate:' in line:
                    rotate_angle = int(line.split(':')[1].strip())
                elif 'Orientation confidence:' in line:
                    confidence = float(line.split(':')[1].strip())
            
            debug_print(f"OSD at {test_angle}°: needs {rotate_angle}° more rotation, confidence={confidence:.1f}")
            
            # If this orientation says it's already upright (rotate: 0) with good confidence
            if rotate_angle == 0 and confidence > best_confidence:
                best_confidence = confidence
                best_angle = test_angle
                
        except Exception as e:
            debug_print(f"OSD failed at {test_angle}°: {e}")
            continue
    
    debug_print(f"Best upright orientation: {best_angle}° (confidence={best_confidence:.1f})")
    return best_angle


def process_deskew(
    image: np.ndarray,
    *,
    use_clahe: bool = True,
    use_gaussian_blur: bool = True,
    use_canny: bool = True,
    use_morph_gradient: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Main deskew processing pipeline.
    1. Find best quad in any of 4 rotations (by area)
    2. Crop to quad
    3. Find upright orientation by testing OSD on all 4 rotations
    4. Fine-tune skew
    """
    image_area = image.shape[0] * image.shape[1]
    best = {
        'contour': None,
        'angle': 0,
        'score': 0,
        'rotation': None,
        'ratio': 1.0
    }

    # Try all 4 orientations - pick the one with largest valid quad
    for angle in [0, 90, 180, 270]:
        rotated = rotate_img(image, angle)
        ratio = rotated.shape[0] / 500.0
        resized = cv2.resize(rotated, (int(rotated.shape[1] / ratio), 500))

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        if use_clahe:
            gray = apply_clahe(gray)
        if use_gaussian_blur:
            gray = apply_gaussian_blur(gray)
        if use_canny:
            edged = apply_canny(gray)
        else:
            edged = gray
        if use_morph_gradient:
            edged = apply_morphological_gradient(edged)

        contour = detect_largest_quadrilateral(edged, image_area)
        
        if contour is not None:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            debug_print(f"Orientation {angle}°: found quad with area={area}")
            
            if area > best['score']:
                best.update({
                    'contour': contour,
                    'angle': angle,
                    'score': area,
                    'rotation': rotated,
                    'ratio': ratio
                })

    # If no quad found, just try to fix skew on original
    if best['contour'] is None:
        debug_print("No valid document boundary found in any orientation")
        fine_angle = compute_best_skew_angle(image)
        if abs(fine_angle) >= 0.5:
            image = deskew_image(image, fine_angle)
        return image, fine_angle

    # Found a quad - do the crop
    debug_print(f"Best contour found at {best['angle']}°")
    screenCnt = best['contour']
    orig = best['rotation']
    ratio = best['ratio']
    
    scaled_pts = screenCnt.reshape(4, 2) * ratio
    warped, rect, w, h = four_point_transform(orig, scaled_pts)
    
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    margins = detect_text_margins(warped_gray)
    debug_print(f"Margins: {margins}")
    buffered_pts = apply_directional_margin(rect.copy(), margins, orig.shape[1], orig.shape[0])
    final_warp, _, _, _ = four_point_transform(orig, buffered_pts)

    # Find the correct upright orientation by testing OSD on all 4 rotations
    upright_angle = find_upright_orientation(final_warp)
    if upright_angle != 0:
        debug_print(f"Rotating cropped image by {upright_angle}° to make upright")
        final_warp = rotate_img(final_warp, upright_angle)

    # Fine-tune skew angle
    fine_angle = compute_best_skew_angle(final_warp)
    if abs(fine_angle) >= 0.5:
        debug_print(f"Skew angle: {fine_angle:.2f}°")
        final_warp = deskew_image(final_warp, fine_angle)
    else:
        fine_angle = 0.0

    return final_warp, fine_angle

def decode_image(file_bytes: bytes) -> np.ndarray:
    """Decode image bytes to numpy array."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img


def encode_image(img: np.ndarray, format: str = "jpeg") -> str:
    """Encode numpy array to base64 string."""
    if format.lower() in ["jpg", "jpeg"]:
        ext = ".jpg"
    elif format.lower() == "png":
        ext = ".png"
    elif format.lower() == "tiff":
        ext = ".tiff"
    elif format.lower() == "bmp":
        ext = ".bmp"
    else:
        ext = ".jpg"
    
    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Failed to encode image")
    
    return base64.b64encode(buffer).decode('utf-8')


def draw_highlight_box(img: np.ndarray, color=(0, 255, 0), thickness: int = 5) -> np.ndarray:
    """Draw a green box around the detected receipt/document."""
    h, w = img.shape[:2]
    # Inset the rectangle slightly so the border is fully visible
    inset = max(thickness, 5)
    top_left = (inset, inset)
    bottom_right = (w - inset, h - inset)
    cv2.rectangle(img, top_left, bottom_right, color, thickness)
    return img


def encode_image_bytes(img: np.ndarray, format: str = "jpeg") -> bytes:
    """Encode numpy array to raw image bytes."""
    if format.lower() in ["jpg", "jpeg"]:
        ext = ".jpg"
    elif format.lower() == "png":
        ext = ".png"
    elif format.lower() == "tiff":
        ext = ".tiff"
    elif format.lower() == "bmp":
        ext = ".bmp"
    else:
        ext = ".jpg"

    success, buffer = cv2.imencode(ext, img)
    if not success:
        raise ValueError("Failed to encode image")

    return buffer.tobytes()


def get_mime_type(format: str) -> str:
    """Map internal image format to MIME type."""
    fmt = format.lower()
    if fmt == "jpeg":
        return "image/jpeg"
    elif fmt == "png":
        return "image/png"
    elif fmt == "tiff":
        return "image/tiff"
    elif fmt == "bmp":
        return "image/bmp"
    return "application/octet-stream"


def detect_image_format(filename: str) -> str:
    """Detect image format from filename."""
    ext = filename.lower().split('.')[-1]
    if ext in ["jpg", "jpeg"]:
        return "jpeg"
    elif ext == "png":
        return "png"
    elif ext in ["tif", "tiff"]:
        return "tiff"
    elif ext == "bmp":
        return "bmp"
    else:
        return "jpeg"


# =================== ROUTES ===================
@router.post("/", response_model=DeskewResponse)
async def deskew_endpoint(
    image: UploadFile = File(..., description="Image file to process"),
    return_angle: bool = Form(False, description="Return detected angle"),
    max_angle: float = Form(45.0, description="Maximum angle to correct"),
    background_color: str = Form("white", description="Background fill color"),
    highlight: bool = Form(False, description="If true/1, draw green box around detected receipt"),
    output: str = Form("json", description="Output type: 'json' or 'file'")
):
    """
    Deskew an image by detecting and correcting skew/rotation.
    Supports JPEG, PNG, TIFF, and BMP formats.
    """
    start_time = time.time()

    tesseract = check_tesseract()
    if not tesseract["available"]:
        raise HTTPException(
            status_code=503,
            detail=tesseract.get("error", "Tesseract is not available. Install Tesseract OCR and ensure it is on your PATH (e.g. brew install tesseract on macOS)."),
        )

    try:
        # Read and validate file
        file_bytes = await image.read()
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        if file_size_mb > MAX_IMAGE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"Image size ({file_size_mb:.2f}MB) exceeds maximum ({MAX_IMAGE_SIZE_MB}MB)"
            )
        
        # Detect format
        img_format = detect_image_format(image.filename or "image.jpg")
        
        # Decode image
        try:
            img = decode_image(file_bytes)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image format. Supported: JPEG, PNG, TIFF, BMP. Error: {str(e)}"
            )
        
        # Validate dimensions
        height, width = img.shape[:2]
        if height > MAX_IMAGE_DIMENSION or width > MAX_IMAGE_DIMENSION:
            raise HTTPException(
                status_code=400,
                detail=f"Image dimensions ({width}x{height}) exceed maximum ({MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION})"
            )
        
        # Process deskew
        try:
            processed_img, detected_angle = process_deskew(img)
            corrected = True
        except ValueError as e:
            if "No valid document boundary found" in str(e):
                processed_img = img
                detected_angle = 0.0
                corrected = False
            else:
                raise HTTPException(status_code=400, detail=str(e))
        
        # Optionally highlight the detected receipt/document region
        if highlight and corrected:
            processed_img = draw_highlight_box(processed_img)
        
        # If requested, return raw image file instead of JSON
        if output and output.lower() == "file":
            try:
                image_bytes = encode_image_bytes(processed_img, img_format)
            except ValueError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to encode output image: {str(e)}"
                )

            mime_type = get_mime_type(img_format)
            filename = f"deskewed_output.{img_format}"
            return StreamingResponse(
                io.BytesIO(image_bytes),
                media_type=mime_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        # Default: encode result as base64 JSON
        encoded_img = encode_image(processed_img, img_format)
        processing_time = (time.time() - start_time) * 1000

        return DeskewResponse(
            image=encoded_img,
            detected_angle=detected_angle,
            corrected=corrected,
            format=img_format,
            processing_time_ms=processing_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint for deskew service. Includes Tesseract availability."""
    tesseract = check_tesseract()
    return {
        "service": "deskew",
        "status": "healthy" if tesseract["available"] else "degraded",
        "tesseract": tesseract,
    }
