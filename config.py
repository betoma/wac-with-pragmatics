import configparser
import os
from dotenv import load_dotenv


def load_config_paths():
    load_dotenv()
    # load config file, set up paths, make project-specific imports
    config_path = os.environ.get("VISCONF")
    if not config_path:
        # try default location, if not in environment
        default_path_to_config = "./default.cfg"
        if os.path.isfile(default_path_to_config):
            config_path = default_path_to_config

    assert (
        config_path is not None
    ), "You need to specify the path to the config file via environment variable VISCONF."

    config = configparser.ConfigParser()
    with open(config_path, "r", encoding="utf-8") as f:
        config.read_file(f)

    corpora_base = config.get("DEFAULT", "corpora_base")
    preproc_path = config.get("DSGV-PATHS", "preproc_path")
    dsgv_home = config.get("DSGV-PATHS", "dsgv_home")

    return corpora_base, preproc_path, dsgv_home
