import os
import yaml


def load_from_yaml(file_path, message=None, print_msg=True):
    _file_path = os.path.expanduser(file_path)

    f = open(_file_path, "r")
    result = yaml.safe_load(f)

    if message is None:
        message = f"Loaded Data from '{_file_path}'"

    if print_msg:
        print(message)

    return result


def dump_to_yaml(file_path, data, message=None):
    _file_path = os.path.expanduser(file_path)

    dir_name = os.path.dirname(_file_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    f = open(_file_path, "w")
    yaml.safe_dump(data, f, sort_keys=False)

    if message is None:
        message = f"Saved Data to '{_file_path}'"
    print(message)


def load_secret(key: str):
    return load_from_yaml("./config/secrets.yaml").get(key)
