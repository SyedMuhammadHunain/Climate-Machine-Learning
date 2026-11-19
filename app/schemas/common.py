from enum import Enum


class ProvinceEnum(str, Enum):
    PUNJAB = "Punjab"
    SINDH = "Sindh"
    BALOCHISTAN = "Balochistan"


class CropEnum(str, Enum):
    APPLE = "apple"
    BANANA = "banana"
    COTTON = "cotton"
    DATES = "dates"
    GRAPES = "grapes"
    MANGO = "mango"
    ONION = "onion"
    POMEGRANATE = "pomegranate"
    POTATO = "potato"
    RICE = "rice"
    SUGARCANE = "sugarcane"
    TOMATO = "tomato"
    WHEAT = "wheat"
