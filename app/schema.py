from pydantic import BaseModel, Field
from typing import Optional

class HouseInput(BaseModel):
    # 🔴 REQUIRED (Top impact features)
    Overall_Qual: int = Field(..., alias="Overall Qual")
    Gr_Liv_Area: int = Field(..., alias="Gr Liv Area")
    Neighborhood: str

    # 🟡 OPTIONAL (auto-filled by adapter)
    Garage_Cars: Optional[int] = Field(None, alias="Garage Cars")
    Garage_Area: Optional[int] = Field(None, alias="Garage Area")
    Total_Bsmt_SF: Optional[int] = Field(None, alias="Total Bsmt SF")
    Year_Built: Optional[int] = Field(None, alias="Year Built")
    Year_Remod_Add: Optional[int] = Field(None, alias="Year Remod/Add")
    Full_Bath: Optional[int] = Field(None, alias="Full Bath")
    TotRms_AbvGrd: Optional[int] = Field(None, alias="TotRms AbvGrd")
    Kitchen_Qual: Optional[str] = Field(None, alias="Kitchen Qual")
    Exter_Qual: Optional[str] = Field(None, alias="Exter Qual")

    class Config:
        populate_by_name = True
