import platform
import sys
# requires pywin32 (See requirements.txt. Supports Windows only obviously)
# also recommend FileMeta https://github.com/Dijji/FileMeta to allow tags for PNG
import pythoncom
try:
    from win32com.propsys import propsys
    from win32com.shell import shellcon
except:
    propsys = None
    shellcon = None

import os
import argparse
import glob

script_path = os.path.dirname(os.path.realpath(__file__))

def set_property(file="", property="System.Keywords", values=[], remove_values=[], remove_all=False, ps=None):
    array_properties = ["System.Keywords", "System.Category"]
    if len(values) == 0 and len(remove_values) == 0 and not remove_all:
        return ps
    # get property store for a given shell item (here a file)
    try:
        pk = propsys.PSGetPropertyKeyFromName(property)
    except:
        pythoncom.CoInitialize()
        pk = propsys.PSGetPropertyKeyFromName(property)

    if (ps is None):
        ps = propsys.SHGetPropertyStoreFromParsingName(os.path.realpath(file), None, shellcon.GPS_READWRITE, propsys.IID_IPropertyStore)

    if property in array_properties:
        # read & print existing (or not) property value, System.Keywords type is an array of string
        existingValues = ps.GetValue(pk).GetValue()
        if existingValues == None:
            existingValues = []
        filteredValues = []

        if not remove_all:
            for value in existingValues:
                if value in remove_values:
                    continue
                filteredValues.append(value)

        # build an array of string type PROPVARIANT
        newValue = propsys.PROPVARIANTType(filteredValues + values, pythoncom.VT_VECTOR | pythoncom.VT_BSTR)

        # write property
        ps.SetValue(pk, newValue)
    
    return ps

def tag_files(files_glob="", tags=[], remove_tags=[], remove_all_tags=False, filename="", comment="", categories=[], remove_categories=[], remove_all_categories=False, log_prefix=""):
    if propsys == None or shellcon == None or platform.system() != "Windows":
        return

    if files_glob=="":
        files = [os.path.realpath(filename)]
    else:
        files = glob.glob(files_glob)

    for file in files:
        try:
            ps = set_property(file=file, property="System.Keywords", values=tags,
                              remove_values=remove_tags, remove_all=remove_all_tags)
            ps = set_property(file=file, property="System.Category", values=categories,
                              remove_values=remove_categories, remove_all=remove_all_categories, ps=ps)
            if ps is not None:
                ps.Commit()
        except:
            print(f"{log_prefix}Unable to write tag or category for {file}")

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--files-glob", type=str, default="", help="glob pattern to files to tag", required=True)
parser.add_argument("-t", "--tags", type=str, default="", help="comma separated list of tags", required=False)
parser.add_argument("-r", "--remove-tags", type=str, default="", help="comma separated list of tags to remove", required=False)
parser.add_argument("-c", "--categories", type=str, default="", help="comma separated list of categories add", required=False)
parser.add_argument("-rc", "--remove-categories", type=str, default="", help="comma separated list of categories to remove", required=False)
parser.add_argument("-rt", "--remove-all-tags", action="store_true", default=False, help="remove all tags", required=False)
parser.add_argument("-rac", "--remove-all-categories", action="store_true", default=False, help="remove all tags", required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.tags == "":
        args.tags = []
    else:
        args.tags = args.tags.split(',')

    if args.categories == "":
        args.categories = []
    else:
        args.categories = args.categories.split(',')

    if args.remove_categories == "":
        args.remove_categories = []
    else:
        args.remove_categories = args.remove_categories.split(',')
    
    tag_files(**args.__dict__)