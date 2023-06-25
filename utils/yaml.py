import yaml


def yaml_load(file='data.yaml'):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, 'r') as file:
        yaml_data: dict = yaml.safe_load(file)
    return yaml_data

def yaml_dump(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)