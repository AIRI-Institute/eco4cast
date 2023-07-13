# https://cloud.google.com/docs/authentication/application-default-credentials
# Put it near notebook file

import time
from smartscheduler.master_machine.compute.client_library.snippets.instances.get import get_instance
from smartscheduler.master_machine.compute.client_library.snippets.disks.autodelete_change import set_disk_autodelete
from smartscheduler.master_machine.compute.client_library.snippets.snapshots.create import create_snapshot
from smartscheduler.master_machine.compute.client_library.snippets.instances.delete import delete_instance
from smartscheduler.master_machine.compute.client_library.snippets.disks.delete import delete_disk
from smartscheduler.master_machine.compute.client_library.snippets.disks.create_from_snapshot import (
    create_disk_from_snapshot,
)
from smartscheduler.master_machine.compute.client_library.snippets.instances.create_start_instance.create_with_existing_disks import (
    create_instance,
    get_disk,
    compute_v1,
)
from smartscheduler.master_machine.compute.client_library.snippets.snapshots.delete import delete_snapshot


def google_cloud_move_vm(
    current_zone,
    current_instance_name,
    project_id,
    new_zone,
    new_instance_name=None,
) -> compute_v1.Instance:
    '''
    Function that makes a snapshot of an instance, copies it to another Google Cloud zone. 
    After that original instance is deleted and new one is created in new zone. 
    '''

    start_time = time.time()
    if new_instance_name is None:
        new_instance_name = current_instance_name

    instance = get_instance(
        project_id=project_id, zone=current_zone, instance_name=current_instance_name
    )

    machine_type = instance.machine_type.split("/")[-1]
    disk = instance.disks[0]

    set_disk_autodelete(
        project_id=project_id,
        zone=current_zone,
        instance_name=current_instance_name,
        disk_name=disk.device_name,
        autodelete=False,
    )

    snapshot_name = f"{disk.device_name}-snapshot"
    snapshot = create_snapshot(
        project_id=project_id,
        disk_name=disk.device_name,
        snapshot_name=snapshot_name,
        zone=current_zone,
    )

    delete_instance(
        project_id=project_id, zone=current_zone, machine_name=current_instance_name
    )

    delete_disk(project_id=project_id, zone=current_zone, disk_name=disk.device_name)

    new_disk = create_disk_from_snapshot(
        project_id=project_id,
        zone=new_zone,
        disk_name=disk.device_name,
        disk_size_gb=disk.disk_size_gb,
        disk_type=f"zones/{new_zone}/diskTypes/pd-standard",
        snapshot_link=snapshot.self_link,
    )


    def create_with_existing_disks(
        project_id: str,
        zone: str,
        instance_name: str,
        disk_names: list[str],
        machine_type: str = "n1-standard-1",
    ) -> compute_v1.Instance:
        """
        Create a new VM instance using selected disks. The first disk in disk_names will
        be used as boot disk.

        Args:
            project_id: project ID or project number of the Cloud project you want to use.
            zone: name of the zone to create the instance in. For example: "us-west3-b"
            instance_name: name of the new virtual machine (VM) instance.
            disk_names: list of disk names to be attached to the new virtual machine.
                First disk in this list will be used as the boot device.

        Returns:
            Instance object.
        """
        assert len(disk_names) >= 1
        disks = [get_disk(project_id, zone, disk_name) for disk_name in disk_names]
        attached_disks = []
        for disk in disks:
            adisk = compute_v1.AttachedDisk()
            adisk.source = disk.self_link
            attached_disks.append(adisk)
        attached_disks[0].boot = True
        instance = create_instance(
            project_id,
            zone,
            instance_name,
            attached_disks,
            machine_type=machine_type,
            external_access=True,
        )
        return instance

    new_instance = create_with_existing_disks(
        project_id=project_id,
        zone=new_zone,
        instance_name=new_instance_name,
        disk_names=[new_disk.name],
        machine_type=machine_type,
    )

   
    delete_snapshot(project_id=project_id, snapshot_name=snapshot_name)

    end_time = time.time()
    end_time - start_time

    return new_instance
