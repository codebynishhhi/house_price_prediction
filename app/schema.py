from pydantic import BaseModel

class HouseInput(BaseModel):
    MS_SubClass: int
    Lot_Frontage: float | None
    Lot_Area: int
    Overall_Qual: int
    Overall_Cond: int
    Year_Built: int
    Year_Remod_Add: int
    Yr_Sold: int
    Neighborhood: str
    Kitchen_Qual: str
    Exter_Qual: str
    Full_Bath: int
    Half_Bath: int
    Bsmt_Full_Bath: int
    Bsmt_Half_Bath: int
    Garage_Cars: int
    Garage_Area: int
    Gr_Liv_Area: int
    TotRms_AbvGrd: int
    ...
