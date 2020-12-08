import argparse
import configargparse
import  sys
import pyproj
import geopandas as gpd
from shapely.geometry import Point, LineString, MultiPoint, Polygon, LinearRing, mapping
from shapely.ops import split, nearest_points
import fiona
import logging
import numpy as np
import  pandas as pd
import os
from shapely.ops import transform
import rasterstats as rs
import rasterio as rio
from osgeo import gdal
import rioxarray as rxr
import pygeoprocessing.geoprocessing as geop
import  math
from mining_trucks.utils import grouped_parallel_apply

from matplotlib import pyplot as plt

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))

parser = configargparse.ArgParser(default_config_files=['configs/gis.yaml'], ignore_unknown_config_file_keys=True)
parser.add_argument('-dm', '--dem_masked', help='smoothing window in meters')
parser.add_argument('-dv', '--dem_vector_points', help='')
parser.add_argument('-qm', '--quality_map', help='')
parser.add_argument('-qf', '--quality_file', help='')
parser.add_argument('-rp', '--roads_profiles_file', help='')
parser.add_argument('-s', '--span_length', help='')
parser.add_argument('-wp', '--main_projection', help='')
parser.add_argument('-mp', '--metric_projection', help='')
parser.add_argument('-n', '--null_height', help='')
parser.add_argument('-sb', '--use_subspans', help='')


args = parser.parse_args(sys.argv[1:])


project = pyproj.Transformer.from_proj(
    pyproj.Proj(init= args.metric_projection ), # source coordinate system
    pyproj.Proj(init= args.main_projection))

def break_line_max_length(line, dist):
    if line.length <= dist:
        return [line]
    else:
        segments = cut(line, dist)
        return [segments[0]] + break_line_max_length(segments[1], dist)

def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # https://shapely.readthedocs.io/en/stable/manual.html
    if distance <= 0.0 or distance >= line.length:
        return [line]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]


def cut_into_spans(dfs_shp_all, skip_roads=[]):
    logging.info("Разбивка дорог по спанам")
    road_spans_df = []  #
    dfs_shp_all = dfs_shp_all[~dfs_shp_all['name'].isin(skip_roads)]
    for road in dfs_shp_all['name']:

        sel_road_df = dfs_shp_all[dfs_shp_all['name'] == road]
        if sel_road_df['direction'].iloc[0] == True:
            temp_line = sel_road_df.iloc[0]['geometry']
        else:
            temp_line = sel_road_df.iloc[0]['geometry']
            reversed_l = list(temp_line.coords)[::-1]
            temp_line = LineString(reversed_l)

        line_segments = break_line_max_length(temp_line, int(args.span_length))

        # Переводим линию в геодатафрейм
        t = gpd.GeoDataFrame(line_segments)
        t['geometry'] = t[0]
        t['len'] = t.length
        t = t.drop([0], axis=1)
        t.crs = args.metric_projection
        mp2 = t.to_crs('epsg:4326')
        mp2 = mp2.reset_index().rename(columns={'index': 'myspan'})
        mp2['road_id'] = road
        road_spans_df.append(mp2)  #
    road_spans_df = pd.concat(road_spans_df).reset_index().drop(['index'], axis=1)
    return  road_spans_df


def spans_to_rectangles(base_speed_df_xy):
    logging.info("Перевод спанов в прямоугольники")
    all_roads_rectangles = []
    all_roads_rectangles_subspans = []

    for road in base_speed_df_xy.road_id.unique():
        base_speed_df_tmp_pr = base_speed_df_xy[base_speed_df_xy.road_id == road]
        df_rect = []
        span_rect_wgs_df = []
        for span_num in base_speed_df_tmp_pr.myspan:

            # 1. Берем спан конкретной дороги
            myline = base_speed_df_tmp_pr.loc[base_speed_df_tmp_pr.myspan == span_num].geometry[
                base_speed_df_tmp_pr[base_speed_df_tmp_pr.myspan == span_num].index[0]]

            # 2. Считаем проекции первой и последней точки спана на границу дороги
            max_index = len(myline.coords.xy[0]) - 1

            l1 = Point(myline.coords.xy[0][0], myline.coords.xy[1][0])
            l2 = Point(myline.coords.xy[0][max_index], myline.coords.xy[1][max_index])

            span_line = LineString([l1, l2])
            span_line_wgs = transform(project.transform, span_line)
            # Разбиваем на спаны на прямоугольники
            left = span_line.parallel_offset(10, 'left')
            right = span_line.parallel_offset(10, 'right')
            span_coords = [left.boundary[0], left.boundary[1], right.boundary[0], right.boundary[1]]
            span_rect = LinearRing(span_coords)
            span_rect_wgs = transform(project.transform, span_rect)
            span_rect_wgs = Polygon(span_rect_wgs)
            span_rect_wgs_dict = {'geometry': span_rect_wgs}
            span_rect_wgs_dict_dfd = pd.Series(span_rect_wgs_dict).to_frame('geometry')
            span_rect_wgs_dict_dfd['span'] = span_num
            span_rect_wgs_df.append(span_rect_wgs_dict_dfd)

            # Разбиваем каждый спан на продольные прямоугольники
            temp_df_width = []
            step = 0.5
            for width in np.arange(-10, 10, step):
                left_start = span_line.parallel_offset(width, 'left')
                if width >= 0:
                    left_end = left_start.parallel_offset(step, 'left')
                    coords = [left_start.boundary[0], left_start.boundary[1], left_end.boundary[1],
                              left_end.boundary[0]]
                else:
                    left_end = left_start.parallel_offset(step, 'right')
                    coords = [left_start.boundary[0], left_start.boundary[1], left_end.boundary[0],
                              left_end.boundary[1]]
                rect = LinearRing(coords)
                r_wgs3 = transform(project.transform, rect)
                r_wgs3 = Polygon(r_wgs3)
                dict_list1 = {'geometry': r_wgs3}
                temp_df_width.append(pd.Series(dict_list1).to_frame('geometry'))

            r_wgs1 = pd.concat(temp_df_width)
            r_wgs1 = gpd.GeoDataFrame(r_wgs1, geometry='geometry')
            r_wgs1.crs = "EPSG:4326"
            r_wgs1['span'] = span_num
            df_rect.append(r_wgs1)

        df_rect = pd.concat(df_rect).reset_index()  # измельченный датафрейм со спанами

        df_rect = df_rect.reset_index().rename(columns={'level_0': 'subspan'})
        df_rect['road_id'] = road

        span_rect_wgs_df = pd.concat(span_rect_wgs_df)
        span_rect_wgs_df = gpd.GeoDataFrame(span_rect_wgs_df, geometry='geometry')
        span_rect_wgs_df['road_id'] = road
        span_rect_wgs_df.crs = "EPSG:4326"  # укрупненный датафрейм по спанам

        all_roads_rectangles_subspans.append(df_rect)
        all_roads_rectangles.append(span_rect_wgs_df)
    all_roads_rectangles_subspans = pd.concat(all_roads_rectangles_subspans)
    all_roads_rectangles = pd.concat(all_roads_rectangles)

    return  all_roads_rectangles, df_rect

def crop_raster(all_roads_rectangles, o_file):
    logging.info("Обрезка растра")
    # Обрезка растра по прямоугольникам
    lidar_chm_im = rxr.open_rasterio(o_file,
                                     masked=True).squeeze()
    # Визуализация исходного растра
    f, ax = plt.subplots(figsize=(10, 10))
    lidar_chm_im.plot(ax=ax)
    ax.set(title="Исходный растр")
    ax.set_axis_off()
    f.savefig("data/processed/initial_raster.png")
#    plt.show()

    lidar_clipped = lidar_chm_im.rio.clip(all_roads_rectangles.geometry.apply(mapping))

    # Сохранение обрезанного растра
    lidar_clipped.rio.to_raster(args.dem_masked)
    # Визуализация обрезанного растра
    f, ax = plt.subplots(figsize=(10, 10))
    lidar_clipped.plot(ax=ax)
    ax.set(title="Обрезанный растр")
    ax.set_axis_off()
    f.savefig("data/processed/cropped_raster.png")

def prepare_rasters(rb):
    logging.info("Подготовка растра высот")
    info_raster = geop.get_raster_info(args.dem_masked)
    cols = info_raster['raster_size'][0]
    rows = info_raster['raster_size'][1]
    geotransform = info_raster['geotransform']
    xsize = geotransform[1]
    ysize = -geotransform[5]
    xmin = geotransform[0]
    ymin = geotransform[3]

    # create one-dimensional arrays for x and y
    x = np.linspace(xmin + xsize / 2, xmin + xsize / 2 + (cols - 1) * xsize, cols)
    y = np.linspace(ymin - ysize / 2, ymin - ysize / 2 - (rows - 1) * ysize, rows)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))

    # Расчет точек непосредственно по изображению без кропа по shp-файлу
    coords = zip(X, Y)
    shapely_points1 = [(point[0], point[1]) for point in coords]

    gt = info_raster['geotransform']

    values = [rb.ReadAsArray(int((point[0] - gt[0]) / gt[1]),  # x pixel
                             int((point[1] - gt[3]) / gt[5]),  # y pixel
                             1, 1)[0][0]
              for point in shapely_points1]

    # creation of the resulting shapefile
    schema = {'geometry': 'Point', 'properties': {'id': 'int', 'value': 'float'}, }
    crs = info_raster['projection_wkt']
    with fiona.open(args.dem_vector_points, 'w', 'ESRI Shapefile', schema, crs)  as output:
        for i, point in enumerate(shapely_points1):
            if (values[i] != float(args.null_height)):
                output.write({'geometry': mapping(Point(point)), 'properties': {'id': i, 'value': str(values[i])}})


def z_from_raster(plot_buffer_path,dtm_pre_arr, sjer_chm_meta):
    point_stats = rs.zonal_stats(plot_buffer_path,
               dtm_pre_arr,
               nodata=-999,
               affine=sjer_chm_meta['transform'],
               geojson_out=True,
               copy_properties=True,
               stats="count min mean max median"
                )
    z_value = point_stats[0]['properties']['mean']
    return z_value


def calc_span_slopes(road_spans_df):
    logging.info("Расчет углов")

    with rio.open(args.dem_masked) as dem_src:
        dtm_pre_arr = dem_src.read(1, masked=True)
        sjer_chm_meta = dem_src.profile

    # Получим координаты первой и последней точки спана
    road_spans_df['first_span_point'] = road_spans_df['geometry'].apply(lambda x: x.boundary[0])
    road_spans_df['last_span_point'] = road_spans_df['geometry'].apply(lambda x: x.boundary[1])

    road_spans_df['h_start'] = road_spans_df['first_span_point'].apply(lambda x: z_from_raster(x, dtm_pre_arr, sjer_chm_meta))
    road_spans_df['h_end'] = road_spans_df['last_span_point'].apply(lambda x: z_from_raster(x, dtm_pre_arr, sjer_chm_meta))

    # Расчитываем уклон спана
    road_spans_df['H_full'] = road_spans_df['h_end'] - road_spans_df['h_start']
    road_spans_df['tan'] = road_spans_df['H_full'] / 20
    road_spans_df['slope_corr'] = road_spans_df['tan'].apply(lambda x: math.degrees(math.atan(x)))
    road_spans_df['slope_corr_rad'] = road_spans_df['tan'].apply(lambda x: math.atan(x))
    return  road_spans_df

def finalize_profiles(road_spans_df):
    logging.info("Извлечение конечных профилей")
    road_local_df = []
    for index, row in road_spans_df.iterrows():  # пробегаемся по всем линейным объектам в shp-файле
        df = pd.DataFrame(row.geometry.coords[:], columns=['X', 'Y'])
        x = df['X']
        y = df['Y']
        road_tmp_xy = pd.DataFrame({'lon': x, 'lat': y})
        #     layer_name  = row['road_id']
        road_tmp_xy['span'] = row[0]
        road_tmp_xy['slope'] = row['slope_corr']
        road_tmp_xy['road'] = row['road_id']
        road_tmp_xy['restored_height'] = row['h_start']
        #     road_tmp_xy['order'] = list(range(len(road_tmp_xy)))
        road_local_df.append(road_tmp_xy)
    road_local_df = pd.concat(road_local_df)

    road_local_gdf = gpd.GeoDataFrame(road_local_df, geometry=gpd.points_from_xy(road_local_df.lon, road_local_df.lat))
    def cumdistancesum(df):
        df.crs = args.main_projection
        df = df.to_crs(args.metric_projection)
        df['distance'] = df.distance(df.shift(-1))
        df['current_distance'] = df['distance'].cumsum()
        df = df.to_crs(args.main_projection)
        return df

    roads_profiles = grouped_parallel_apply(road_local_gdf.groupby('road'), cumdistancesum).reset_index(drop=True)
    rp = []
    for p in roads_profiles.road.unique():
        profile = roads_profiles.loc[roads_profiles.road==p,:]
        profile.loc[profile.restored_height <= profile.restored_height.quantile(0.01),'restored_height'] = None
        profile['restored_height'] = profile['restored_height'].ffill().bfill()
        profile = profile.sort_values('current_distance').reset_index()
        if profile.iloc[-1]['restored_height'] < profile.iloc[0]['restored_height']:
            profile['current_distance'] = profile['current_distance'].max() - profile['current_distance']
        rp.append(profile)
    roads_profiles = pd.concat(rp)
    return roads_profiles[['lat', 'lon', 'span', 'road', 'restored_height', 'current_distance']]

def calc_span_height_std(all_proc_data, road_spans_df,all_roads_rectangles,df_rect):
    logging.info("Считаем std высот на спанах")
    # Переводим абсолютную высоту в относительную (относительно спана)
    new_df = all_proc_data.to_crs(args.main_projection)
    final_df = new_df.merge(road_spans_df, left_on=['road_id', 'span'], right_on=['road_id', 'myspan'], )
    final_df['H_related'] = final_df.tan * final_df.projection_distance_metric
    final_df['corrected_height'] = final_df.value - final_df.H_related

    if args.use_subspans == 'False':
        sel_final_df = final_df[['corrected_height', 'span', 'road_id']]
        final_gdf = all_roads_rectangles.merge(sel_final_df, on=['road_id', 'span'])
        res = final_gdf.dissolve(by=['road_id', 'span'], aggfunc='std').rename(
            columns={'corrected_height': 'height_std'}).reset_index()

    else:
        sel_final_df = final_df[['corrected_height', 'span', 'subspan', 'road_id']]
        final_gdf = all_roads_rectangles.merge(sel_final_df, on=['road_id', 'span'])
        # Расчет  std по подспанам
        final_gdf_subspan = df_rect.merge(sel_final_df, on=['road_id', 'span', 'subspan'])
        res2 = final_gdf_subspan.dissolve(by='subspan', aggfunc={'corrected_height': 'std', 'span': 'first'}).rename(
            columns={'corrected_height': 'height_std'})
        # Расчет  std по спанам усреднением по подспанам
        res = res2.dissolve(by=['road_id', 'span'], aggfunc='mean').reset_index()
    res = res.rename(columns={"road_id":"road"})
    return res


def profile_from_geo(ortho_file_name, roads_file_name):
    road_elg = gpd.read_file(roads_file_name)
    road_elg = road_elg.rename(columns={'id': 'name'})
    # #Все дороги карьера
    dfs_shp_all = road_elg
    # # Переход к плоской СК
    dfs_shp_all.crs = args.main_projection
    dfs_shp_all.to_crs(args.metric_projection, inplace=True)
    # # Меняем направление дороги, следим чтобы она совпадала с направлением Погрузка - Разгрузка
    dfs_shp_all['direction'] = False  # По умалчанию все дороги направлением Погрузка-разгрузка
    road_spans_df = cut_into_spans(dfs_shp_all)
    base_speed_df_xy = road_spans_df.to_crs(args.metric_projection)  #
    all_roads_rectangles, df_rect = spans_to_rectangles(base_speed_df_xy)
    crop_raster(all_roads_rectangles, ortho_file_name)
    src_ds = gdal.Open(args.dem_masked)
    rb = src_ds.GetRasterBand(1)
    prepare_rasters(rb)
    road_spans_df = calc_span_slopes(road_spans_df)
    roads_profiles = pd.DataFrame(finalize_profiles(road_spans_df))
    gdf_matrix_new = gpd.read_file(args.dem_vector_points)
    sjoined_points = gpd.sjoin(gdf_matrix_new , all_roads_rectangles , how="inner", op='intersects')
    sjoined_points_merged =  sjoined_points
    gdf_roads_metric = road_spans_df.to_crs(args.metric_projection)
    # Перевод телеметрии в геодатафрейм и метрическую проекцию
    sjoined_points_merged.crs = args.main_projection
    sjoined_points_merged_metric = sjoined_points_merged.to_crs(args.main_projection)

    def projection_gdf1(df):
        temp_road_gdf = gdf_roads_metric[
            (gdf_roads_metric.road_id == df.road_id.iloc[0]) & (gdf_roads_metric.myspan == df.span.iloc[0])]
        df['projection_distance_metric'] = df.geometry.apply(lambda x: temp_road_gdf.project(Point(x.x, x.y)))
        return df

    all_proc_data = []
    for road in gdf_roads_metric.road_id.unique():
        temp_df = sjoined_points_merged_metric[sjoined_points_merged_metric.road_id == road]
        processed_data = grouped_parallel_apply(temp_df.groupby('span'), projection_gdf1).reset_index(drop=True)
        all_proc_data.append(processed_data)
    all_proc_data = pd.concat(all_proc_data)
    h_std = pd.DataFrame(calc_span_height_std(all_proc_data, road_spans_df,all_roads_rectangles,df_rect))
    roads_profiles = pd.DataFrame(roads_profiles.merge(h_std, on=['road','span']))
    height_std = pd.Series(np.array(roads_profiles['height_std'])).astype(float)
    lat = pd.Series(np.array(roads_profiles['lat'])).astype(float)
    lon = pd.Series(np.array(roads_profiles['lon'])).astype(float)
    current_distance = pd.Series(np.array(roads_profiles['current_distance'])).astype(float)
    restored_height = pd.Series(np.array(roads_profiles['restored_height'])).astype(float)
    span = pd.Series(np.array(roads_profiles['span'])).astype(int)
    road = pd.Series(roads_profiles['road'])

    roads_profiles = pd.DataFrame({"road":road, "span": span, "lat": lat, "lon": lon,
                                   "current_distance": current_distance, "restored_height": restored_height,
                                   "height_std": height_std})

    return roads_profiles

