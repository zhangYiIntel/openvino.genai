# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino.properties.hint as hints
import openvino.properties as props
import openvino as ov
import os
from optimum.intel.openvino.utils import TemporaryDirectory
from pathlib import Path


def get_default_llm_properties():
    return {
        hints.inference_precision: ov.Type.f32,
        hints.kv_cache_precision: ov.Type.f16,
    }


def extra_generate_kwargs():
    from optimum.intel.utils.import_utils import is_transformers_version
    additional_args = {}
    if is_transformers_version(">=", "4.51"):
        additional_args["use_model_defaults"] = False

    return additional_args


OV_MODEL_FILENAME = "openvino_model.xml"
OV_TOKENIZER_FILENAME = "openvino_tokenizer.xml"
OV_DETOKENIZER_FILENAME = "openvino_detokenizer.xml"


def get_disabled_mmap_ov_config():
    return {props.enable_mmap: False}


def get_ov_cache_models_dir(temp_dir=TemporaryDirectory()):
    ov_cache = os.environ.get("OV_CACHE", temp_dir.name)
    return Path(ov_cache) / "test_models"
