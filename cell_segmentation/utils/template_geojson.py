# -*- coding: utf-8 -*-
# GeoJson templates
#
# @ Fabian Hörst, fabian.hoerst@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen


def get_template_point() -> dict:
    """Return a template for a Point geojson object

    Returns:
        dict: Template
    """
    template_point = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPoint",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_point


def get_template_segmentation() -> dict:
    """Return a template for a MultiPolygon geojson object

    Returns:
        dict: Template
    """
    template_multipolygon = {
        "type": "Feature",
        "id": "TODO",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": [
                [],
            ],
        },
        "properties": {
            "objectType": "annotation",
            "classification": {"name": "TODO", "color": []},
        },
    }
    return template_multipolygon
