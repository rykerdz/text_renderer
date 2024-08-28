# All Effect/Layout example config
# 1. Run effect_layout_example.py, generate images in effect_layout_image
# 2. Update README.md
import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    NormPerspectiveTransformCfg,
    GeneratorCfg,
    SimpleTextColorCfg,
    TextColorCfg,
    FixedTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.effect.curve import Curve
from text_renderer.layout import SameLineLayout, ExtraTextLineLayout

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
OUT_DIR = CURRENT_DIR / "output"
DATA_DIR = CURRENT_DIR
BG_DIR = DATA_DIR / "bg"
FONT_DIR = DATA_DIR / "font"
FONT_LIST_DIR = DATA_DIR / "font_list"
TEXT_DIR = DATA_DIR / "corpus"

font_cfg = dict(
    font_dir=FONT_DIR,
    font_list_file=FONT_LIST_DIR / "font_list.txt",
    font_size=(30, 31),
    num_word=(1,3),
)


def base_cfg(
    name: str, corpus, corpus_effects=None, layout_effects=None, layout=None, gray=True
):
    return GeneratorCfg(
        num_image=5,
        save_dir=OUT_DIR / name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            perspective_transform=perspective_transform,
            gray=gray,
            layout_effects=layout_effects,
            layout=layout,
            corpus=corpus,
            corpus_effects=corpus_effects,
        ),
    )


def dropout_rand():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    return cfg

def fiji_word_data():
    return base_cfg(
        inspect.currentframe().f_code.co_name,
        corpus=WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "fiji_text.txt"],
                **font_cfg
            ),
        ),
    )

def dropout_horizontal():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    return cfg


def dropout_vertical():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    return cfg


def line():
    poses = [
        "top",
        "bottom",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]
    cfgs = []
    for i, pos in enumerate(poses):
        pos_p = [0] * len(poses)
        pos_p[i] = 1
        cfg = base_cfg(f"{inspect.currentframe().f_code.co_name}_{pos}")
        cfg.render_cfg.corpus_effects = Effects(
            Line(p=1, thickness=(3, 4), line_pos_p=pos_p)
        )
        cfgs.append(cfg)
    return cfgs


def padding():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True)
    )
    return cfg




def extra_text_line_layout():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.layout = ExtraTextLineLayout(bottom_prob=1.0)
    cfg.render_cfg.corpus = [
        WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
        WordCorpus(
            WordCorpusCfg(
                text_paths=[TEXT_DIR / "eng_text.txt"],
                filter_by_chars=True,
                chars_file=CHAR_DIR / "eng.txt",
                **font_cfg
            ),
        ),
    ]
    return cfg


def color_image():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.gray = False
    return cfg


def perspective_transform():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.perspective_transform = FixedPerspectiveTransformCfg(30, 30, 1.5)
    return cfg


def compact_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = -0.3
    return cfg


def large_char_spacing():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus.cfg.char_spacing = 0.5
    return cfg


def curve():
    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            Curve(p=1, period=180, amplitude=(4, 5)),
        ]
    )
    return cfg



def emboss():
    import imgaug.augmenters as iaa

    cfg = base_cfg(inspect.currentframe().f_code.co_name)
    cfg.render_cfg.height = 48
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1, w_ratio=[0.2, 0.21], h_ratio=[0.7, 0.71], center=True),
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    return cfg


configs = [
    emboss()
    extra_text_line_layout()
    char_spacing_compact(),
    char_spacing_large(),
    *line(),
    perspective_transform(),
    dropout_rand(),
    dropout_horizontal(),
    dropout_vertical(),
    padding(),
]
