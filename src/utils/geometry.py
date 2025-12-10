import math
from typing import Optional, Tuple

Point = Tuple[float, float]


def angle_between_points(a: Point, b: Point, c: Point) -> Optional[float]:
    """
    Compute the angle ABC (in degrees) given three points:
      A (ax, ay), B (bx, by), C (cx, cy)

    Angle is at point B, between BA and BC.

    Returns:
        Angle in degrees, or None if the points are degenerate.
    """
    (ax, ay), (bx, by), (cx, cy) = a, b, c

    # Vector BA = A - B
    bax = ax - bx
    bay = ay - by

    # Vector BC = C - B
    bcx = cx - bx
    bcy = cy - by

    # Compute dot product and magnitudes
    dot = bax * bcx + bay * bcy
    mag_ba = math.sqrt(bax**2 + bay**2)
    mag_bc = math.sqrt(bcx**2 + bcy**2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    # Clamp cosine to [-1, 1] to avoid numerical issues
    cos_angle = dot / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)
    return angle_deg
