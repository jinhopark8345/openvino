# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# ov_runtime.dll directory path visibility is needed to use _pyngraph module
# import below causes adding this path to os.environ["PATH"]
import openvino  # noqa: F401 'imported but unused'
