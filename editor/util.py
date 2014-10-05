from math import pow


def clamp(value, minvalue, maxvalue):
    return max(minvalue, min(maxvalue, value))


def toSRGB(value):
    if value <= 0.0031308:
        return clamp(int(255 * 12.92 * value), 0, 255)
    return clamp(255 * (1.055 * pow(value, 1.0/2.4) - 0.055), 0, 255)


def fromSRGB(value):
    if value <= 0.04045:
        return value / 255.0 * (1.0 / 12.92)
    return pow((value / 255.0 + 0.055) * (1.0 / 1.055), 2.4)
