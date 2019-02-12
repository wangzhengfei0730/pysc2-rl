from pysc2.lib import features

_SCREEN_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_SCREEN_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_MINIMAP_PLAYER_ID = features.MINIMAP_FEATURES.player_id.index


def minimap_channel():
    minimap_channel = 0
    for i in range(len(features.MINIMAP_FEATURES)):
        if i == _MINIMAP_PLAYER_ID:
            minimap_channel += 1
        elif features.MINIMAP_FEATURES[i].type == features.FeatureType.SCALAR:
            minimap_channel += 1
        else:
            minimap_channel += features.MINIMAP_FEATURES[i].scale
    return minimap_channel


def screen_channel():
    screen_channel = 0
    for i in range(len(features.SCREEN_FEATURES)):
        if i == _SCREEN_PLAYER_ID or i == _SCREEN_UNIT_TYPE:
            screen_channel += 1
        elif features.SCREEN_FEATURES[i].type == features.FeatureType.SCALAR:
            screen_channel += 1
        else:
            screen_channel += features.SCREEN_FEATURES[i].scale
    return screen_channel
