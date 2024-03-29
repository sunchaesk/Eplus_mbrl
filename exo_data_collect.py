
import sys

import base as base

import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

import io
from urllib.request import urlopen, Request
import pandas as pd

import itertools

def read_epw(filename, coerce_year=None):
    r'''
    Read an EPW file in to a pandas dataframe.

    Note that values contained in the metadata dictionary are unchanged
    from the EPW file.

    EPW files are commonly used by building simulation professionals
    and are widely available on the web. For example via:
    https://energyplus.net/weather , http://climate.onebuilding.org or
    http://www.ladybug.tools/epwmap/


    Parameters
    ----------
    filename : String
        Can be a relative file path, absolute file path, or url.

    coerce_year : None or int, default None
        If supplied, the year of the data will be set to this value. This can
        be a useful feature because EPW data is composed of data from
        different years.
        Warning: EPW files always have 365*24 = 8760 data rows;
        be careful with the use of leap years.


    Returns
    -------
    data : DataFrame
        A pandas dataframe with the columns described in the table
        below. For more detailed descriptions of each component, please
        consult the EnergyPlus Auxiliary Programs documentation [1]_

    metadata : dict
        The site metadata available in the file.

    See Also
    --------
    pvlib.iotools.parse_epw

    Notes
    -----

    The returned structures have the following fields.

    ===============   ======  =========================================
    key               format  description
    ===============   ======  =========================================
    loc               String  default identifier, not used
    city              String  site loccation
    state-prov        String  state, province or region (if available)
    country           String  site country code
    data_type         String  type of original data source
    WMO_code          String  WMO identifier
    latitude          Float   site latitude
    longitude         Float   site longitude
    TZ                Float   UTC offset
    altitude          Float   site elevation
    ===============   ======  =========================================


    +-------------------------------+-----------------------------------------+
    | EPWData field                 | description                             |
    +===============================+=========================================+
    | index                         | A pandas datetime index. NOTE, times are|
    |                               | set to local standard time (daylight    |
    |                               | savings is not included). Days run from |
    |                               | 0-23h to comply with PVLIB's convention.|
    +-------------------------------+-----------------------------------------+
    | year                          | Year, from original EPW file. Can be    |
    |                               | overwritten using coerce function.      |
    +-------------------------------+-----------------------------------------+
    | month                         | Month, from original EPW file.          |
    +-------------------------------+-----------------------------------------+
    | day                           | Day of the month, from original EPW     |
    |                               | file.                                   |
    +-------------------------------+-----------------------------------------+
    | hour                          | Hour of the day from original EPW file. |
    |                               | Note that EPW's convention of 1-24h is  |
    |                               | not taken over in the index dataframe   |
    |                               | used in PVLIB.                          |
    +-------------------------------+-----------------------------------------+
    | minute                        | Minute, from original EPW file. Not     |
    |                               | used.                                   |
    +-------------------------------+-----------------------------------------+
    | data_source_unct              | Data source and uncertainty flags. See  |
    |                               | [1]_, chapter 2.13                      |
    +-------------------------------+-----------------------------------------+
    | temp_air                      | Dry bulb temperature at the time        |
    |                               | indicated, deg C                        |
    +-------------------------------+-----------------------------------------+
    | temp_dew                      | Dew-point temperature at the time       |
    |                               | indicated, deg C                        |
    +-------------------------------+-----------------------------------------+
    | relative_humidity             | Relative humidity at the time indicated,|
    |                               | percent                                 |
    +-------------------------------+-----------------------------------------+
    | atmospheric_pressure          | Station pressure at the time indicated, |
    |                               | Pa                                      |
    +-------------------------------+-----------------------------------------+
    | etr                           | Extraterrestrial horizontal radiation   |
    |                               | recv'd during 60 minutes prior to       |
    |                               | timestamp, Wh/m^2                       |
    +-------------------------------+-----------------------------------------+
    | etrn                          | Extraterrestrial normal radiation recv'd|
    |                               | during 60 minutes prior to timestamp,   |
    |                               | Wh/m^2                                  |
    +-------------------------------+-----------------------------------------+
    | ghi_infrared                  | Horizontal infrared radiation recv'd    |
    |                               | during 60 minutes prior to timestamp,   |
    |                               | Wh/m^2                                  |
    +-------------------------------+-----------------------------------------+
    | ghi                           | Direct and diffuse horizontal radiation |
    |                               | recv'd during 60 minutes prior to       |
    |                               | timestamp, Wh/m^2                       |
    +-------------------------------+-----------------------------------------+
    | dni                           | Amount of direct normal radiation       |
    |                               | (modeled) recv'd during 60 minutes prior|
    |                               | to timestamp, Wh/m^2                    |
    +-------------------------------+-----------------------------------------+
    | dhi                           | Amount of diffuse horizontal radiation  |
    |                               | recv'd during 60 minutes prior to       |
    |                               | timestamp, Wh/m^2                       |
    +-------------------------------+-----------------------------------------+
    | global_hor_illum              | Avg. total horizontal illuminance recv'd|
    |                               | during the 60 minutes prior to          |
    |                               | timestamp, lx                           |
    +-------------------------------+-----------------------------------------+
    | direct_normal_illum           | Avg. direct normal illuminance recv'd   |
    |                               | during the 60 minutes prior to          |
    |                               | timestamp, lx                           |
    +-------------------------------+-----------------------------------------+
    | diffuse_horizontal_illum      | Avg. horizontal diffuse illuminance     |
    |                               | recv'd during the 60 minutes prior to   |
    |                               | timestamp, lx                           |
    +-------------------------------+-----------------------------------------+
    | zenith_luminance              | Avg. luminance at the sky's zenith      |
    |                               | during the 60 minutes prior to          |
    |                               | timestamp, cd/m^2                       |
    +-------------------------------+-----------------------------------------+
    | wind_direction                | Wind direction at time indicated,       |
    |                               | degrees from north (360 = north; 0 =    |
    |                               | undefined,calm)                         |
    +-------------------------------+-----------------------------------------+
    | wind_speed                    | Wind speed at the time indicated, m/s   |
    +-------------------------------+-----------------------------------------+
    | total_sky_cover               | Amount of sky dome covered by clouds or |
    |                               | obscuring phenomena at time stamp,      |
    |                               | tenths of sky                           |
    +-------------------------------+-----------------------------------------+
    | opaque_sky_cover              | Amount of sky dome covered by clouds or |
    |                               | obscuring phenomena that prevent        |
    |                               | observing the sky at time stamp, tenths |
    |                               | of sky                                  |
    +-------------------------------+-----------------------------------------+
    | visibility                    | Horizontal visibility at the time       |
    |                               | indicated, km                           |
    +-------------------------------+-----------------------------------------+
    | ceiling_height                | Height of cloud base above local terrain|
    |                               | (7777=unlimited), meter                 |
    +-------------------------------+-----------------------------------------+
    | present_weather_observation   | Indicator for remaining fields: If 0,   |
    |                               | then the observed weather codes are     |
    |                               | taken from the following field. If 9,   |
    |                               | then missing weather is assumed.        |
    +-------------------------------+-----------------------------------------+
    | present_weather_codes         | Present weather code, see [1], chapter  |
    |                               | 2.9.1.28                                |
    +-------------------------------+-----------------------------------------+
    | precipitable_water            | Total precipitable water contained in a |
    |                               | column of unit cross section from earth |
    |                               | to top of atmosphere, cm. Note that some|
    |                               | old \*_TMY3.epw files may have incorrect|
    |                               | unit if it was retrieved from           |
    |                               | www.energyplus.net.                     |
    +-------------------------------+-----------------------------------------+
    | aerosol_optical_depth         | The broadband aerosol optical depth per |
    |                               | unit of air mass due to extinction by   |
    |                               | aerosol component of atmosphere,        |
    |                               | unitless                                |
    +-------------------------------+-----------------------------------------+
    | snow_depth                    | Snow depth in centimeters on the day    |
    |                               | indicated, (999 = missing data)         |
    +-------------------------------+-----------------------------------------+
    | days_since_last_snowfall      | Number of days since last snowfall      |
    |                               | (maximum value of 88, where 88 = 88 or  |
    |                               | greater days; 99 = missing data)        |
    +-------------------------------+-----------------------------------------+
    | albedo                        | The ratio of reflected solar irradiance |
    |                               | to global horizontal irradiance,        |
    |                               | unitless                                |
    +-------------------------------+-----------------------------------------+
    | liquid_precipitation_depth    | The amount of liquid precipitation      |
    |                               | observed at indicated time for the      |
    |                               | period indicated in the liquid          |
    |                               | precipitation quantity field,           |
    |                               | millimeter                              |
    +-------------------------------+-----------------------------------------+
    | liquid_precipitation_quantity | The period of accumulation for the      |
    |                               | liquid precipitation depth field, hour  |
    +-------------------------------+-----------------------------------------+


    References
    ----------

    .. [1] `EnergyPlus documentation, Auxiliary Programs
       <https://energyplus.net/documentation>`_
    '''

    if str(filename).startswith('http'):
        # Attempts to download online EPW file
        # See comments above for possible online sources
        request = Request(filename, headers={'User-Agent': (
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_5) '
            'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 '
            'Safari/537.36')})
        response = urlopen(request)
        with io.StringIO(response.read().decode(errors='ignore')) as csvdata:
            data, meta = parse_epw(csvdata, coerce_year)

    else:
        # Assume it's accessible via the file system
        with open(str(filename), 'r') as csvdata:
            data, meta = parse_epw(csvdata, coerce_year)


    return data, meta



def parse_epw(csvdata, coerce_year=None):
    """
    Given a file-like buffer with data in Energy Plus Weather (EPW) format,
    parse the data into a dataframe.

    Parameters
    ----------
    csvdata : file-like buffer
        a file-like buffer containing data in the EPW format

    coerce_year : None or int, default None
        If supplied, the year of the data will be set to this value. This can
        be a useful feature because EPW data is composed of data from
        different years.
        Warning: EPW files always have 365*24 = 8760 data rows;
        be careful with the use of leap years.

    Returns
    -------
    data : DataFrame
        A pandas dataframe with the columns described in the table
        below. For more detailed descriptions of each component, please
        consult the EnergyPlus Auxiliary Programs documentation
        available at: https://energyplus.net/documentation.

    metadata : dict
        The site metadata available in the file.

    See Also
    --------
    pvlib.iotools.read_epw
    """
    # Read line with metadata
    firstline = csvdata.readline()

    head = ['loc', 'city', 'state-prov', 'country', 'data_type', 'WMO_code',
            'latitude', 'longitude', 'TZ', 'altitude']
    meta = dict(zip(head, firstline.rstrip('\n').split(",")))

    meta['altitude'] = float(meta['altitude'])
    meta['latitude'] = float(meta['latitude'])
    meta['longitude'] = float(meta['longitude'])
    meta['TZ'] = float(meta['TZ'])

    colnames = ['year', 'month', 'day', 'hour', 'minute', 'data_source_unct',
                'temp_air', 'temp_dew', 'relative_humidity',
                'atmospheric_pressure', 'etr', 'etrn', 'ghi_infrared', 'ghi',
                'dni', 'dhi', 'global_hor_illum', 'direct_normal_illum',
                'diffuse_horizontal_illum', 'zenith_luminance',
                'wind_direction', 'wind_speed', 'total_sky_cover',
                'opaque_sky_cover', 'visibility', 'ceiling_height',
                'present_weather_observation', 'present_weather_codes',
                'precipitable_water', 'aerosol_optical_depth', 'snow_depth',
                'days_since_last_snowfall', 'albedo',
                'liquid_precipitation_depth', 'liquid_precipitation_quantity']

    # We only have to skip 6 rows instead of 7 because we have already used
    # the realine call above.
    data = pd.read_csv(csvdata, skiprows=6, header=0, names=colnames)

    # Change to single year if requested
    if coerce_year is not None:
        data["year"] = coerce_year

    # create index that supplies correct date and time zone information
    dts = data[['month', 'day']].astype(str).apply(lambda x: x.str.zfill(2))
    hrs = (data['hour'] - 1).astype(str).str.zfill(2)
    dtscat = data['year'].astype(str) + dts['month'] + dts['day'] + hrs
    idx = pd.to_datetime(dtscat, format='%Y%m%d%H')
    idx = idx.dt.tz_localize(int(meta['TZ'] * 3600))
    data.index = idx

    data['minute'] = data['minute'].replace(0, pd.NaT).fillna(method='ffill').fillna(0).astype(int)

    return data, meta

OUTDOOR_TEMP = 0 # exo
DIRECT_SOLAR = 6
SITE_HORIZONTAL_INFRARED = 7
OUTDOOR_RELATIVE_HUMIDITY = 8
YEAR = 9
HOUR = 10
DAY_OF_WEEK = 11
DAY = 12
MONTH = 13


default_args = {'idf': './in.idf',
                'epw': './weather.epw',
                'csv': True,
                'output': './output',
                'timesteps': 1000000.0,
                'num_workers': 2,
                'annual': False,# for some reasons if not annual, funky results
                }

def collect_data():
    # outdoor_temp_data = []
    # direct_solar_data = []
    # horizontal_infrared_data = []
    # outdoor_relative_humidity_data = []
    outdoor_temp_data = []
    direct_solar_data = []
    horizontal_infrared_data = []
    outdoor_relative_humidity_data = []
    time_data = []

    env = base.EnergyPlusEnv(default_args)
    state = env.reset()
    done = False


    while not done:
        action = 0 # collecting exo states that are not dependent on action (indoor setpoints)

        ret = n_state, reward, done, truncated, info = env.step(action)

        outdoor_temp_data.append(n_state[OUTDOOR_TEMP])
        direct_solar_data.append(n_state[DIRECT_SOLAR])
        horizontal_infrared_data.append(n_state[SITE_HORIZONTAL_INFRARED])
        outdoor_relative_humidity_data.append(n_state[OUTDOOR_RELATIVE_HUMIDITY])
        # outdoor_temp_data.append(info['obs_vec'][OUTDOOR_TEMP])
        # direct_solar_data.append(info['obs_vec'][DIRECT_SOLAR])
        # horizontal_infrared_data.append(info['obs_vec'][HORIZONTAL_INFRARED])
        # outdoor_relative_humidity_data.append(info['obs_vec'][OUTDOOR_RELATIVE_HUMIDITY])

        # if info['minute'] > 60:
        #     print('MINUTE', info['minute'])
        #     print("HIT")
        #     sys.exit()

        current_time = [
            n_state[YEAR],
            n_state[MONTH],
            n_state[DAY],
            n_state[HOUR]
            # info['year'],
            # info['month'],
            # info['day'],
            # info['hour'],
            #(info['minute'] // 10) * 10
        ]

        time_data.append(current_time)

        state = n_state

    sequence = [10,20,30,40,50,60]
    repeated_sequence = itertools.cycle(sequence)

    temp_cnt = 0
    temp_i = 0
    while time_data[temp_i][3] == 0:
        temp_cnt += 1
        temp_i += 1

    for i in range(6 - temp_cnt):
        next(repeated_sequence)

    for i in range(len(time_data)):
        time_data[i].append(next(repeated_sequence))
        time_data[i] = tuple(time_data[i])

    ret_dict = dict()
    for i in range(len(time_data)):
        ret_dict[time_data[i]] = {
            'outdoor_temp': outdoor_temp_data[i],
            'site_direct_solar': direct_solar_data[i],
            'site_horizontal_infrared': horizontal_infrared_data[i],
            'outdoor_relative_humidity': outdoor_relative_humidity_data[i]
        }

    ret_dict['run_period'] = tuple([time_data[0], time_data[-1]])
    #print(ret_dict)

    save = True# Currently saved from 6-1 ~ 7-15
    if save:
        with open('./exo_state.pt', 'wb') as handle:
            pickle.dump(ret_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('SAVED!')

def _collect_data():
    ret = read_epw('../weather.epw')
    data = ret[0]
    for i in range(10):
        print('hour', data.iloc[i]['hour'], 'minute', data.iloc[i]['minute'])

if __name__ == "__main__":
    #_collect_data()
    collect_data()
