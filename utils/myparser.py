import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.safe_load(fo.read()))
        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.safe_load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def get_config(config_file=None):
    return YamlParser(config_file=config_file)

def getYamlConfig(config_yml_file, configList_yml_file=None):
    cfg = get_config()
    cfg.merge_from_file(config_file=config_yml_file,)
    if configList_yml_file is not None:
        cfg.merge_from_file(config_file=configList_yml_file)
    return cfg