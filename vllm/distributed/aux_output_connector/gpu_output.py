# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput
    from vllm.v1.worker.gpu.async_utils import AsyncOutput


def finish_aux_output(output: AsyncOutput, pending: PendingAuxOutput) -> None:
    """Commit R3 state before the GPU runner starts its next execution step."""
    output.copy_event.synchronize()
    output.model_runner_output.aux_output_connector_output = (
        pending.connector.process_output(
            output.model_runner_output.req_ids,
            pending.token_starts,
            pending.query_start_loc,
            output.routed_experts,
            output.num_sampled_tokens_np,
            output.num_rejected,
        )
    )
