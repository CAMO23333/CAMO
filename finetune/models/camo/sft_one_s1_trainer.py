from .lora_one_s1_trainer import CAMOS1Trainer
from ..utils import register

class CAMOS1SFTTrainer(CAMOS1Trainer):
    pass

register("camo", "sft", CAMOS1SFTTrainer)
