from pydantic import BaseModel


class PredictorRequest(BaseModel):
    tipo_correo: str
    pais: str
    ciudad: str
