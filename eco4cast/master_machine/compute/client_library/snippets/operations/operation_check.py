#  Copyright 2022 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# flake8: noqa


# This file is automatically generated. Please do not modify it directly.
# Find the relevant recipe file in the samples/recipes or samples/ingredients
# directory and apply your changes there.


# [START compute_instances_operation_check]
from google.cloud import compute_v1


def wait_for_operation(
    operation: compute_v1.Operation, project_id: str
) -> compute_v1.Operation:
    """
    This method waits for an operation to be completed. Calling this function
    will block until the operation is finished.

    Args:
        operation: The Operation object representing the operation you want to
            wait on.
        project_id: project ID or project number of the Cloud project you want to use.

    Returns:
        Finished Operation object.
    """
    kwargs = {"project": project_id, "operation": operation.name}
    if operation.zone:
        client = compute_v1.ZoneOperationsClient()
        # Operation.zone is a full URL address of a zone, so we need to extract just the name
        kwargs["zone"] = operation.zone.rsplit("/", maxsplit=1)[1]
    elif operation.region:
        client = compute_v1.RegionOperationsClient()
        # Operation.region is a full URL address of a region, so we need to extract just the name
        kwargs["region"] = operation.region.rsplit("/", maxsplit=1)[1]
    else:
        client = compute_v1.GlobalOperationsClient()
    return client.wait(**kwargs)


# [END compute_instances_operation_check]
