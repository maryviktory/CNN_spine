import ast
import numpy as np
from xml.etree.ElementTree import ElementTree
import os

resource_package = __file__
xml_file_dir = os.path.join(os.path.split(resource_package)[0], "xml_files")


def remove_property(root, field):
    for rank in root.iter("property"):
        if rank.get("name") == field[0]:

            if len(field) == 1:
                root.remove(rank)
            else:
                remove_property(rank, field[1::])


def add_property(root, field_name, field_root):
    for rank in root.iter("property"):
        if rank.get("name") == field_name[0]:

            if len(field_name) == 1:
                rank.append(field_root)
                return
            else:
                add_property(rank, field_name[1::], field_root)


def get_param(root, property_tree, param_name):
    param_val = "ciao"
    for rank in root.iter("property"):
        if rank.get("name") == property_tree[0]:

            if len(property_tree) == 1:
                for param in rank.iter("param"):
                    if param.get("name") == param_name:
                        return param.text
            else:
                param_val = get_param(rank, property_tree[1::], param_name)

    return param_val


def extract_splines_from_ws(root):
    transducer_splines = []
    for prop in root.iter("property"):
        if "GlSpline" not in prop.get("name"):
            continue

        for param in prop.iter("param"):
            if "points" not in param.get("name"):
                continue
            transducer_splines.append(get_array_from_spline_string(param.text))
    return transducer_splines


def extract_label_from_ws(root):
    transducer_splines = []
    for prop in root.iter("property"):
        if "GlPoint" not in prop.get("name"):
            continue

        for param in prop.iter("param"):
            if "points" not in param.get("name"):
                continue
            transducer_splines.append(get_array_from_spline_string(param.text))
    return transducer_splines


def get_array_from_spline_string(ugly_string, decimals=3):
    tmp_string = ugly_string.replace("\n", " ")
    tmp_string = ','.join(tmp_string.split())
    tmp_array = ast.literal_eval("[" + tmp_string[0:-1] + "]")
    spline_array = np.around(tmp_array, decimals=decimals)
    return spline_array


def get_spline_string_from_array(spline_array):
    tmp = str(spline_array)[1:-1]
    spline_string = ' '.join(tmp.split())
    return spline_string


def get_xml_root(object_name):

    if object_name == "ultrasound_hybrid_simulation":
        xml_filename = "ultrasound_simulation_structure.xml"

    elif object_name == "export_file":
        xml_filename = "export_file_structure.xml"

    elif object_name == "spline":
        xml_filename = "spline_structure.xml"
    else:
        return None

    tree = ElementTree()
    tree.parse(os.path.join(xml_file_dir, xml_filename))
    root = tree.getroot()
    return root
