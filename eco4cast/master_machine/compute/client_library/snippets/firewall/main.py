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


from __future__ import annotations

from collections.abc import Iterable

from google.cloud import compute_v1


def create_firewall_rule(
    project_id: str, firewall_rule_name: str, network: str = "global/networks/default"
) -> compute_v1.Firewall:
    """
    Creates a simple firewall rule allowing for incoming HTTP and HTTPS access from the entire Internet.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
        firewall_rule_name: name of the rule that is created.
        network: name of the network the rule will be applied to. Available name formats:
            * https://www.googleapis.com/compute/v1/projects/{project_id}/global/networks/{network}
            * projects/{project_id}/global/networks/{network}
            * global/networks/{network}

    Returns:
        A Firewall object.
    """
    firewall_rule = compute_v1.Firewall()
    firewall_rule.name = firewall_rule_name
    firewall_rule.direction = "INGRESS"

    allowed_ports = compute_v1.Allowed()
    allowed_ports.I_p_protocol = "tcp"
    allowed_ports.ports = ["80", "443"]

    firewall_rule.allowed = [allowed_ports]
    firewall_rule.source_ranges = ["0.0.0.0/0"]
    firewall_rule.network = network
    firewall_rule.description = "Allowing TCP traffic on port 80 and 443 from Internet."

    firewall_rule.target_tags = ["web"]

    # Note that the default value of priority for the firewall API is 1000.
    # If you check the value of `firewall_rule.priority` at this point it
    # will be equal to 0, however it is not treated as "set" by the library and thus
    # the default will be applied to the new rule. If you want to create a rule that
    # has priority == 0, you need to explicitly set it so:
    # TODO: Uncomment to set the priority to 0
    # firewall_rule.priority = 0

    firewall_client = compute_v1.FirewallsClient()
    operation = firewall_client.insert(
        project=project_id, firewall_resource=firewall_rule
    )

    wait_for_extended_operation(operation, "firewall rule creation")

    return firewall_client.get(project=project_id, firewall=firewall_rule_name)


def delete_firewall_rule(project_id: str, firewall_rule_name: str) -> None:
    """
    Deletes a firewall rule from the project.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
        firewall_rule_name: name of the firewall rule you want to delete.
    """
    firewall_client = compute_v1.FirewallsClient()
    operation = firewall_client.delete(project=project_id, firewall=firewall_rule_name)

    wait_for_extended_operation(operation, "firewall rule deletion")


def get_firewall_rule(project_id: str, firewall_rule_name: str) -> compute_v1.Firewall:
    """
    Retrieve a Firewall from a project.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
        firewall_rule_name: name of the firewall rule you want to retrieve.

    Returns:
        A Firewall object.
    """
    firewall_client = compute_v1.FirewallsClient()
    return firewall_client.get(project=project_id, firewall=firewall_rule_name)


def list_firewall_rules(project_id: str) -> Iterable[compute_v1.Firewall]:
    """
    Return a list of all the firewall rules in specified project. Also prints the
    list of firewall names and their descriptions.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.

    Returns:
        A flat list of all firewall rules defined for given project.
    """
    firewall_client = compute_v1.FirewallsClient()
    firewalls_list = firewall_client.list(project=project_id)

    for firewall in firewalls_list:
        print(f" - {firewall.name}: {firewall.description}")

    return firewalls_list


def patch_firewall_priority(
    project_id: str, firewall_rule_name: str, priority: int
) -> None:
    """
    Modifies the priority of a given firewall rule.

    Args:
        project_id: project ID or project number of the Cloud project you want to use.
        firewall_rule_name: name of the rule you want to modify.
        priority: the new priority to be set for the rule.
    """
    firewall_rule = compute_v1.Firewall()
    firewall_rule.priority = priority

    # The patch operation doesn't require the full definition of a Firewall object. It will only update
    # the values that were set in it, in this case it will only change the priority.
    firewall_client = compute_v1.FirewallsClient()
    operation = firewall_client.patch(
        project=project_id, firewall=firewall_rule_name, firewall_resource=firewall_rule
    )

    wait_for_extended_operation(operation, "firewall rule patching")


if __name__ == "__main__":
    import google.auth
    import google.auth.exceptions

    try:
        default_project_id = google.auth.default()[1]
        print(f"Using project {default_project_id}.")
    except google.auth.exceptions.DefaultCredentialsError:
        print(
            "Please use `gcloud auth application-default login` "
            "or set GOOGLE_APPLICATION_CREDENTIALS to use this script."
        )
    else:
        import uuid

        rule_name = "firewall-sample-" + uuid.uuid4().hex[:10]
        print(f"Creating firewall rule {rule_name}...")
        # The rule will be created with default priority of 1000.
        create_firewall_rule(default_project_id, rule_name)
        try:
            print("Rule created:")
            print(get_firewall_rule(default_project_id, rule_name))
            print("Updating rule priority to 10...")
            patch_firewall_priority(default_project_id, rule_name, 10)
            print("Rule updated: ")
            print(get_firewall_rule(default_project_id, rule_name))
            print(f"Deleting rule {rule_name}...")
        finally:
            delete_firewall_rule(default_project_id, rule_name)
        print("Done.")
