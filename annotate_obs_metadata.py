import numpy as np
from astropy.table import Table
from astropy.table import Column
from astropy.table import vstack
from astropy.table import join
import astropy.coordinates as ascoord
import sys
import datetime as dt
import bossdata.path as bdpath
import bossdata.remote as bdremote
import bossdata.spec as bdspec

from progressbar import ProgressBar, Percentage, Bar

ephemeris_block_size_minutes = 15
ephemeris_block_size = dt.timedelta(minutes=ephemeris_block_size_minutes)
ephemeris_max_block_size = dt.timedelta(minutes=1.5*ephemeris_block_size_minutes)

def find_ephemeris_lookup_date(tai_beg, tai_end, obs_md_table):
    '''
    Want to find the 15-minute increment (0, 15, 30, 45) that is sandwiched between the two
    passed datetimes.  However, spans can be less than 15 minutes, so also need handle this
    case; here, will round to whichever increment has the smallest delta between tai_beg
    and tai_end.  I've not seen it, but I have to assume that spans can also be longer than
    15 minutes.
    '''
    vectfunc = np.vectorize(tai_str_to_datetime)
    tai_end_dt = vectfunc(tai_end)
    tai_beg_dt = vectfunc(tai_beg)

    vectfunc = np.vectorize(get_ephemeris_block_in_interval)

    mask = (tai_end_dt - tai_beg_dt) <= ephemeris_max_block_size
    ret = np.zeros((len(tai_end_dt),), dtype=dt.datetime)
    ret[mask] = vectfunc(tai_beg_dt[mask], tai_end_dt[mask])

    def _lookup_str_format(dtval):
        if isinstance(dtval, dt.datetime):
            return dtval.strftime("%Y-%b-%d %H:%M")
        return ""

    vectfunc = np.vectorize(_lookup_str_format)
    ret = vectfunc(ret)

    return ret[mask], obs_md_table[mask]

def get_ephemeris_block_in_interval(tai_beg, tai_end):
    tai_end_dt = tai_str_to_datetime(tai_end)
    tai_beg_dt = tai_str_to_datetime(tai_beg)
    tai_beg_block = round_tai(tai_beg_dt)
    tai_end_block = round_tai(tai_end_dt)

    if tai_beg_block < tai_end_dt  and  tai_beg_block >= tai_beg_dt:
        return tai_beg_block
    elif tai_end_block < tai_end_dt  and  tai_end_block >= tai_beg_dt:
        return tai_end_block
    else:
        end_delta = get_tai_block_delta(tai_end_dt, "down")
        beg_delta = get_tai_block_delta(tai_beg_dt, "up")
        if abs(beg_delta) < abs(end_delta):
            return tai_beg_dt + beg_delta
        else:
            return tai_end_dt + end_delta

def tai_str_to_datetime(tai):
    if isinstance(tai, basestring):
        if tai.count(':') == 2:
            if tai.count('.') == 1:
                tai = dt.datetime.strptime(tai, "%Y-%m-%d %H:%M:%S.%f")
                tai = tai.replace(microsecond=0)
            else:
                tai = dt.datetime.strptime(tai, "%Y-%m-%d %H:%M:%S")
        elif tai.count(':') == 1:
            tai = dt.datetime.strptime(tai, "%Y-%m-%d %H:%M")

    return tai

def get_tai_block_delta(tai, direction="closest"):
    tai = tai_str_to_datetime(tai)

    delta_mins = - ((tai.minute % ephemeris_block_size_minutes) + (tai.second / 60.0))
    if direction == 'closest':
        if delta_mins <= -ephemeris_block_size_minutes/2:
            delta_mins = ephemeris_block_size_minutes + delta_mins
    elif direction == 'down':
        #nada
        True
    elif direction == 'up':
        delta_mins = ephemeris_block_size_minutes + delta_mins

    return dt.timedelta(minutes=delta_mins)

def round_tai(tai):
    tai = tai_str_to_datetime(tai)
    tdelta = get_tai_block_delta(tai)

    return tai + tdelta

def main():
    obs_md_table = Table.read(sys.argv[1], format="ascii")
    lunar_md_table = Table.read(sys.argv[2], format="ascii")
    lunar_md_table.rename_column('UTC', 'EPHEM_DATE')
    solar_md_table = Table.read(sys.argv[3], format="ascii")
    solar_md_table.rename_column('UTC', 'EPHEM_DATE')

    print "Table has {} entries".format(len(obs_md_table))
    lookup_date, obs_md_table = find_ephemeris_lookup_date(obs_md_table['TAI-BEG'], obs_md_table['TAI-END'], obs_md_table)
    print "Successfully got {} ephemeris date entries".format(len(lookup_date))
    ephem_date_col = Column(lookup_date, name="EPHEM_DATE")
    obs_md_table.add_column(ephem_date_col)

    def _get_separation(start, end, cache=None):
        if cache is not None:
            if cache.has_key((start,end)):
                return cache.get((start,end))
            else:
                val = start.separation(end).degree
                cache[(start,end)] = val
                return val
        return start.separation(end).degree

    separation_vectfunc = np.vectorize(_get_separation)

    boresight_ra_dec = ascoord.SkyCoord(ra=obs_md_table['RA'], dec=obs_md_table['DEC'], unit='deg', frame='fk5')
    galactic_core = ascoord.SkyCoord(l=0.0, b=0.0, unit='deg', frame='galactic')

    def _get_galactic_plane_separation(start, cache=None):
        if cache is not None:
            if cache.has_key(start):
                return cache.get(start)
            else:
                val = start.transform_to('galactic').b.degree
                cache[start] = val
                return val
        return start.transform_to('galactic').b.degree

    plane_separation_vectfunc = np.vectorize(_get_galactic_plane_separation)

    #Join lunar data to the table
    obs_md_table = join(obs_md_table, lunar_md_table['EPHEM_DATE', 'RA_APP', 'DEC_APP', 'MG_APP', 'ELV_APP'])
    #print obs_md_table
    cache = {}
    obs_md_table.add_column(Column(separation_vectfunc(boresight_ra_dec,
                                        ascoord.SkyCoord(ra=obs_md_table['RA_APP'], dec=obs_md_table['DEC_APP'], unit='deg', frame='icrs'),
                                        cache),
                                    dtype=float, name="LUNAR_SEP"))

    obs_md_table.rename_column("MG_APP", "LUNAR_MAGNITUDE")
    obs_md_table.rename_column("ELV_APP", "LUNAR_ELV")
    obs_md_table.remove_columns(['RA_APP', 'DEC_APP'])
    #print obs_md_table

    #Join solar data to the table
    obs_md_table = join(obs_md_table, solar_md_table['EPHEM_DATE', 'RA_APP', 'DEC_APP', 'ELV_APP'])
    #print obs_md_table
    cache = {}
    obs_md_table.add_column(Column(separation_vectfunc(boresight_ra_dec,
                                        ascoord.SkyCoord(ra=obs_md_table['RA_APP'], dec=obs_md_table['DEC_APP'], unit='deg', frame='icrs'),
                                        cache),
                                    dtype=float, name="SOLAR_SEP"))
    obs_md_table.rename_column("ELV_APP", "SOLAR_ELV")
    obs_md_table.remove_columns(['RA_APP', 'DEC_APP'])
    #print obs_md_table

    #Add in galactic data
    #Room to improve performance here; since same RA/DEC for many exposures, could
    #cache galactic core, plane separations.
    cache = {}
    obs_md_table.add_column(Column(separation_vectfunc(boresight_ra_dec, galactic_core, cache),
                                dtype=float, name="GALACTIC_CORE_SEP"))
    cache = {}
    obs_md_table.add_column(Column(plane_separation_vectfunc(boresight_ra_dec, cache),
                                dtype=float, name="GALACTIC_PLANE_SEP"))
    #print obs_md_table

    obs_md_table.write("annnotated_metadata.csv", format="ascii.csv")

if __name__ == '__main__':
    main()
