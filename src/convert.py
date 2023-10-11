import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil
from glob import glob
from tqdm import tqdm
import json
import imagesize


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def create_ann(image_path):
    filename = sly.fs.get_file_name_with_ext(image_path)
    (
        height,
        width,
    ) = imagesize.get(image_path)
    labels = []

    if filename in data:
        for tag_value, bbox in data[filename]:
            xmin, ymin, xmax, ymax = bbox
            rectangle = sly.Rectangle(int(ymin), int(xmin), int(ymax), int(xmax))
            tag = sly.Tag(tm_character, tag_value)
            label = sly.Label(rectangle, class_character, tags=[tag])
            labels.append(label)

    return sly.Annotation(img_size=(height, width), labels=labels)


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


class_character = sly.ObjClass("character", sly.Rectangle, color=[255, 0, 0])
tm_character = sly.TagMeta("character", sly.TagValueType.ANY_STRING)


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    project = api.project.create(workspace_id, project_name)
    meta = sly.ProjectMeta(obj_classes=list(class_character), tag_metas=[tm_character])
    api.project.update_meta(project.id, meta.to_json())

    dataset_path = "/mnt/c/users/german/documents/CDD"
    dir_paths = ["test", "train", "val"]

    batch_size = 50
    for ds in dir_paths:
        global data
        dspath = os.path.join(dataset_path, "dataset", ds)
        dataset = api.dataset.create(project.id, ds, change_name_if_conflict=True)
        images_pathes = glob(os.path.join(dspath, "*"))
        ds_ann_path = os.path.join(
            dataset_path, "annotations", "generated", "{}_annotations.json".format(ds)
        )
        with open(ds_ann_path) as f:
            data = json.load(f)
        progress = sly.Progress("Create dataset {}".format(ds), len(images_pathes))
        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            img_names_batch = [
                sly.fs.get_file_name_with_ext(im_path) for im_path in img_pathes_batch
            ]
            img_infos = api.image.upload_paths(dataset.id, img_names_batch, img_pathes_batch)
            img_ids = [im_info.id for im_info in img_infos]
            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)
    return project
