import json


# hacky way
def to_dict(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.__dict__))


def stringify_obj(obj):
    return json.dumps(obj, default=lambda o: o.__dict__)


def stringify_obj_beautiful(obj):
    return json.dumps(obj, default=lambda o: o.__dict__, indent=4)
