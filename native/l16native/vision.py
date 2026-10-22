"""Apple Vision (Neural Engine) přes PyObjC: maska osob, obličeje, horizont.

Vstup: sRGB obraz uint8 (H, W, 3) v orientaci zobrazení. Výstup: maska 0..1 ve stejném rozměru,
obdélníky obličejů (normalizované, počátek vlevo nahoře) a úhel horizontu.
"""

from __future__ import annotations

from typing import Any

import cv2
import Foundation
import numpy as np
import Quartz
import Vision
from numpy.typing import NDArray

ONE_COMPONENT_8 = 0x4C303038  # kCVPixelFormatType_OneComponent8 ('L008')


def _nsdata_png(rgb8: NDArray[np.uint8]) -> Any:
    ok, png = cv2.imencode(".png", np.ascontiguousarray(rgb8[..., ::-1]))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return Foundation.NSData.dataWithBytes_length_(png.tobytes(), len(png))


def _pixel_buffer_to_array(pb: Any) -> NDArray[np.uint8]:
    Quartz.CVPixelBufferLockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        h = Quartz.CVPixelBufferGetHeight(pb)
        w = Quartz.CVPixelBufferGetWidth(pb)
        stride = Quartz.CVPixelBufferGetBytesPerRow(pb)
        base = Quartz.CVPixelBufferGetBaseAddress(pb)
        buf = base.as_buffer(stride * h)
        arr = np.frombuffer(buf, np.uint8).reshape(h, stride)[:, :w].copy()
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(pb, Quartz.kCVPixelBufferLock_ReadOnly)
    return arr


def analyze(rgb8: NDArray[np.uint8]) -> dict[str, Any]:
    h, w, _ = rgb8.shape
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(_nsdata_png(rgb8), None)
    seg = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
    seg.setQualityLevel_(Vision.VNGeneratePersonSegmentationRequestQualityLevelAccurate)
    seg.setOutputPixelFormat_(ONE_COMPONENT_8)
    faces = Vision.VNDetectFaceRectanglesRequest.alloc().initWithCompletionHandler_(None)
    horizon = Vision.VNDetectHorizonRequest.alloc().initWithCompletionHandler_(None)
    ok, err = handler.performRequests_error_([seg, faces, horizon], None)
    if not ok:
        raise RuntimeError(f"Vision: {err}")

    mask = np.zeros((h, w), np.float32)
    if seg.results():
        m = _pixel_buffer_to_array(seg.results()[0].pixelBuffer()).astype(np.float32) / 255.0
        mask = np.asarray(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)

    face_boxes = []
    for f in faces.results() or []:
        bb = f.boundingBox()  # normalizované, počátek vlevo dole
        x, y, bw, bh = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
        face_boxes.append(
            {"x": x, "y": 1.0 - y - bh, "w": bw, "h": bh, "confidence": float(f.confidence())}
        )

    angle = None
    if horizon.results():
        angle = float(horizon.results()[0].angle())  # radiány, + = proti směru hodin

    return {
        "mask": mask,
        "faces": face_boxes,
        "horizon_rad": angle,
        "person_pct": round(float((mask > 0.5).mean()) * 100, 1),
    }


def upright_score(rgb8: NDArray[np.uint8]) -> float:
    """Kolik „vzpřímených“ lidí obraz obsahuje: obličeje s rollem < 30° + těla, kde krk je nad boky.

    Detektory samy jsou na otočení necitlivé, ale roll obličeje a směr kořen→krk ne.
    """
    import math

    ih, iw = rgb8.shape[:2]
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(_nsdata_png(rgb8), None)
    faces = Vision.VNDetectFaceRectanglesRequest.alloc().initWithCompletionHandler_(None)
    body = Vision.VNDetectHumanBodyPoseRequest.alloc().initWithCompletionHandler_(None)
    handler.performRequests_error_([faces, body], None)
    score = 0.0
    for f in faces.results() or []:
        if f.roll() is not None and abs(math.degrees(float(f.roll()))) < 30:
            score += float(f.confidence())
    for o in body.results() or []:
        pts, _ = o.recognizedPointsForJointsGroupName_error_(
            Vision.VNHumanBodyPoseObservationJointsGroupNameAll, None
        )
        n = pts.get(Vision.VNHumanBodyPoseObservationJointNameNeck)
        r = pts.get(Vision.VNHumanBodyPoseObservationJointNameRoot)
        if n is not None and r is not None and n.confidence() > 0.3 and r.confidence() > 0.3:
            dx = (n.location().x - r.location().x) * iw
            dy = (n.location().y - r.location().y) * ih  # Vision: y roste nahoru
            if abs(math.degrees(math.atan2(dx, dy))) < 35:
                score += 1.0
    return score
