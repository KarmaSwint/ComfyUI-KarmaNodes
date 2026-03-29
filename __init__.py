from .KarmaKSamplerCycle import Karma_KSampler_Cycle
from .KarmaFilmGrain import Karma_Film_Grain
from .KarmaKolors import Karma_Kolors
from .KarmaLensFX import Karma_Lens_FX
from .KarmaToneCurves import Karma_Tone_Curves
from .KarmaFilmEmulation import Karma_Film_Emulation

NODE_CLASS_MAPPINGS = {
    "Karma-KSampler-Cycle": Karma_KSampler_Cycle,
    "Karma-Film-Grain": Karma_Film_Grain,
    "Karma-Kolors": Karma_Kolors,
    "Karma-Lens-FX": Karma_Lens_FX,
    "Karma-Tone-Curves": Karma_Tone_Curves,
    "Karma-Film-Emulation": Karma_Film_Emulation,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Karma-KSampler-Cycle": "Karma KSampler Cycle",
    "Karma-Film-Grain": "Karma Film Grain",
    "Karma-Kolors": "Karma Kolors",
    "Karma-Lens-FX": "Karma Lens FX",
    "Karma-Tone-Curves": "Karma Tone Curves",
    "Karma-Film-Emulation": "Karma Film Emulation",
}
