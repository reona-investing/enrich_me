from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainerOutputs:
    model: object = None
    scaler: Optional[object] = None