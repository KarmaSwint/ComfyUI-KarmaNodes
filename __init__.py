from .KarmaKSamplerCycle import Karma_KSampler_Cycle
from .KarmaFilmGrain import Karma_Film_Grain
from .KarmaKolors import Karma_Kolors

NODE_CLASS_MAPPINGS = {
    "Karma-KSampler-Cycle": Karma_KSampler_Cycle,
    "Karma-Film-Grain": Karma_Film_Grain,
    "Karma-Kolors": Karma_Kolors,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Karma-KSampler-Cycle": "Karma KSampler Cycle",
    "Karma-Film-Grain": "Karma Film Grain",
    "Karma-Kolors": "Karma Kolors",
} 