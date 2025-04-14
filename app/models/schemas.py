from pydantic import BaseModel
from typing import Optional

class LicensePlateResponse(BaseModel):
    license_plate: str

class IDCardResponse(BaseModel):
    full_name: str
    id_number: str

class VehicleDocResponse(BaseModel):
    owner_name: str
    license_plate: str

class OCRResponse(BaseModel):
    license_plate_data: Optional[LicensePlateResponse] = None
    id_card_data: Optional[IDCardResponse] = None
    vehicle_doc_data: Optional[VehicleDocResponse] = None