"""
Geohash processing pipeline for demand-forecasting models (e.g. LightGBM).

Usage:
    from process_geohash import GeohashProcessor

    gp = GeohashProcessor(n_clusters=15)
    train_df = gp.fit_transform(train_df)   # fits KMeans on train geohashes
    test_df  = gp.transform(test_df)        # reuses the same fitted clusters

Requires: pip install pygeohash --break-system-packages
"""

import pandas as pd
import numpy as np
import pygeohash as pgh
from sklearn.cluster import KMeans
import pickle


class GeohashProcessor:
    def __init__(self, n_clusters=15, zone_prefix_len=5, random_state=42):
        self.n_clusters = n_clusters
        self.zone_prefix_len = zone_prefix_len
        self.random_state = random_state
        self.kmeans = None
        self.center_lat = None
        self.center_lon = None
        self._fitted = False

    def _decode(self, df):
        """Decode geohash strings -> continuous lat/lon columns."""
        df = df.copy()
        lat_lon = df['geohash'].apply(pgh.decode)
        df['lat'] = lat_lon.apply(lambda x: x[0])
        df['lon'] = lat_lon.apply(lambda x: x[1])
        return df

    def fit_transform(self, df):
        """Fit KMeans + center on this dataframe (use on TRAIN only), then transform it."""
        df = self._decode(df)

        # coarser regional zone (categorical, low-ish cardinality)
        df['geo_zone'] = df['geohash'].str[:self.zone_prefix_len]

        # city center = centroid of all training locations
        self.center_lat = df['lat'].mean()
        self.center_lon = df['lon'].mean()
        df['dist_to_center'] = np.sqrt(
            (df['lat'] - self.center_lat) ** 2 + (df['lon'] - self.center_lon) ** 2
        )

        # fit KMeans on UNIQUE geohash centroids only (not row-duplicated data),
        # so busy locations don't bias the cluster centers
        uniq = df[['geohash', 'lat', 'lon']].drop_duplicates('geohash')
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        uniq['geo_cluster'] = self.kmeans.fit_predict(uniq[['lat', 'lon']])
        df = df.merge(uniq[['geohash', 'geo_cluster']], on='geohash', how='left')

        self._fitted = True
        return self._finalize(df)

    def transform(self, df):
        """Apply an already-fitted processor to new data (val/test/inference)."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform on training data first.")

        df = self._decode(df)
        df['geo_zone'] = df['geohash'].str[:self.zone_prefix_len]
        df['dist_to_center'] = np.sqrt(
            (df['lat'] - self.center_lat) ** 2 + (df['lon'] - self.center_lon) ** 2
        )

        uniq = df[['geohash', 'lat', 'lon']].drop_duplicates('geohash')
        uniq['geo_cluster'] = self.kmeans.predict(uniq[['lat', 'lon']])
        df = df.merge(uniq[['geohash', 'geo_cluster']], on='geohash', how='left')

        return self._finalize(df)

    def _finalize(self, df):
        # cast to category dtype so LightGBM can use them as native categoricals
        df['geo_zone'] = df['geo_zone'].astype('category')
        df['geo_cluster'] = df['geo_cluster'].astype('category')
        return df

    def save(self, path='geohash_processor.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path='geohash_processor.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    # Example run against train.csv
    df = pd.read_csv('/mnt/user-data/uploads/train.csv')

    gp = GeohashProcessor(n_clusters=15, zone_prefix_len=5)
    df_out = gp.fit_transform(df)

    print(df_out[['geohash', 'lat', 'lon', 'geo_zone', 'dist_to_center', 'geo_cluster']].head(10))
    print("\nNew columns added:", ['lat', 'lon', 'geo_zone', 'dist_to_center', 'geo_cluster'])
    print("\ngeo_cluster value counts:")
    print(df_out['geo_cluster'].value_counts())

    # Save the fitted processor so the SAME clusters/center get reused on test data
    gp.save('/mnt/user-data/outputs/geohash_processor.pkl')

    # Save the processed training data
    df_out.to_csv('/mnt/user-data/outputs/train_with_geo_features.csv', index=False)
    print("\nSaved: train_with_geo_features.csv and geohash_processor.pkl")
