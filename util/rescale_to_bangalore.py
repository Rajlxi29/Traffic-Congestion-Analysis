"""
Rescales synthetic/anonymized geohash-decoded lat/lon values onto Bangalore's
real-world bounding box, so they can be plotted on an actual map in Tableau.

IMPORTANT CAVEAT:
This does NOT recover true coordinates. It's a linear transform that preserves
the RELATIVE spatial structure of your geohash cells (which ones are close to
each other, clustering patterns, relative distances) while remapping them to
sit visually within Bangalore's real geographic extent. Treat the resulting
lat/lon as "display coordinates for visualization," not ground truth.

Usage:
    from rescale_to_bangalore import rescale_to_city_bbox

    df = rescale_to_city_bbox(df, lat_col='lat', lon_col='lon')
    # adds df['display_lat'], df['display_lon']
"""

import pandas as pd

# Approximate real bounding box for Bangalore city + surrounding metro area
BANGALORE_BBOX = {
    'lat_min': 12.83,
    'lat_max': 13.14,
    'lon_min': 77.35,
    'lon_max': 77.78,
}


def rescale_to_city_bbox(df, lat_col='lat', lon_col='lon',
                          target_bbox=BANGALORE_BBOX,
                          preserve_aspect_ratio=True):
    """
    Linearly rescales df[lat_col]/df[lon_col] into target_bbox.

    preserve_aspect_ratio=True (recommended):
        Uses a SINGLE scale factor for both lat and lon, derived from whichever
        axis is more constrained. This keeps relative distances/shapes between
        geohash cells intact -- just shifted and scaled uniformly, like moving
        a printed map rather than stretching it unevenly.

    preserve_aspect_ratio=False:
        Independently stretches lat and lon to fully fill the target bbox.
        Uses the whole box more evenly but distorts relative shapes/distances.
    """
    df = df.copy()

    src_lat_min, src_lat_max = df[lat_col].min(), df[lat_col].max()
    src_lon_min, src_lon_max = df[lon_col].min(), df[lon_col].max()

    src_lat_range = src_lat_max - src_lat_min
    src_lon_range = src_lon_max - src_lon_min

    tgt_lat_range = target_bbox['lat_max'] - target_bbox['lat_min']
    tgt_lon_range = target_bbox['lon_max'] - target_bbox['lon_min']

    if preserve_aspect_ratio:
        # pick the tighter-fitting scale so both axes fit inside the target box
        scale = min(tgt_lat_range / src_lat_range, tgt_lon_range / src_lon_range)
        lat_scale = lon_scale = scale
    else:
        lat_scale = tgt_lat_range / src_lat_range
        lon_scale = tgt_lon_range / src_lon_range

    # normalize source to 0..1, apply scale, center within target bbox
    norm_lat = (df[lat_col] - src_lat_min) / src_lat_range
    norm_lon = (df[lon_col] - src_lon_min) / src_lon_range

    scaled_lat_span = src_lat_range * lat_scale
    scaled_lon_span = src_lon_range * lon_scale

    lat_offset = target_bbox['lat_min'] + (tgt_lat_range - scaled_lat_span) / 2
    lon_offset = target_bbox['lon_min'] + (tgt_lon_range - scaled_lon_span) / 2

    df['display_lat'] = lat_offset + norm_lat * scaled_lat_span
    df['display_lon'] = lon_offset + norm_lon * scaled_lon_span

    return df


if __name__ == '__main__':
    import pygeohash as pgh

    df = pd.read_csv('/mnt/user-data/uploads/train.csv')
    lat_lon = df['geohash'].apply(pgh.decode)
    df['lat'] = lat_lon.apply(lambda x: x[0])
    df['lon'] = lat_lon.apply(lambda x: x[1])

    df = rescale_to_city_bbox(df, preserve_aspect_ratio=True)

    print(df[['geohash', 'lat', 'lon', 'display_lat', 'display_lon']].drop_duplicates('geohash').head(10))
    print('\ndisplay_lat range:', df['display_lat'].min(), df['display_lat'].max())
    print('display_lon range:', df['display_lon'].min(), df['display_lon'].max())
    print('\n(should sit within Bangalore bbox:', BANGALORE_BBOX, ')')
