from dataclasses import dataclass

@dataclass
class TrainerOutputs:
    models: list[object] = None
    scalers: list[object] = None