import lt_toolbox as ltt
import polars as pl
import xarray as xr
from netCDF4 import Dataset as ds
import numpy as np
import f90nml as nl
import fileinput
from os import system
import time
from glob import glob as ls
import gsw
from netCDF4 import num2date
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import datetime as dt
import matplotlib.path as mpath
from numpy.random import randint as ri
import pandas as pd
from tqdm import tqdm

def read_tracmass(project,case,
                  oname='output',ext='out',
                  inpath = '/home/jdo043/DATA/ArMOC/Tracmass/output/',
                  save=False,start_time = '1995-01-01'):
    if project == 'GLORYS12_native':
        if case in ['Nordic8','Nordic9'] or 'daily' in case:
            fz = ds('/home/jdo043/native_GLORYS/daily_test/glorys12v1-daily_corr-gridT_19950201.nc')
        elif case in ['Nordic10','Bering1']:
            fz = ds('/home/jdo043/native_GLORYS/glorys12v1-monthly-gridT_m_199502.nc')
        elif 'Arctic' in case:
            fz = ds('/home/jdo043/native_GLORYS/Arctic/glorys12v1-monthly-gridT_m_199502.nc')
        else:
            fz = ds('/home/jdo043/native_GLORYS/Arctic/glorys12v1-monthly-gridT_m_199502.nc')
            
    elif project == 'ORAS5':
        fa = ds('/home/jdo043/DATA/ORAS5/input/mesh_hgr.nc')
        fz = ds('/home/jdo043/DATA/ORAS5/input/ORAS5_gridT_199502.nc')
    elif project == 'NEMO':
        fz = ds('/home/jdo043/DATA/ORCA_5day_test/ORCA0083-N01_gridT_19951106.nc')
        
    lon = fz.variables['nav_lon'][:] 
    lat = fz.variables['nav_lat'][:]
    z = fz.variables['deptht'][:]
    
    filename = inpath+project+'/'+case+'/'+oname+'_'+ext+'.csv'
    
    if ext in ['out','ini','run']:
        columns = ['id','x','y','z','vt','time','wall','temp','salt','dens']
        dtypes = {d:float for d in columns}; dtypes['id'] = int
    elif ext == 'err':
        columns = ['id','x','y','z','vt','time','err']
        dtypes = {d:float for d in columns}; dtypes['id'] = int; dtypes['err'] = str
    data = pl.read_csv(filename,new_columns=columns,has_header=False,dtypes=dtypes,null_values=['********','****************'])

    if ext != 'err':
        data = data.filter(pl.col("salt") > 0.)
    len_y = lat.shape[0]

    
    data = data.filter(pl.col("y") < len_y);
    data = data.filter(pl.col('z').is_finite())
    
    T = ltt.TrajFrame(source=data,condense=True);
    T = T.transform_trajectory_coords(lon = lon,lat = lat,depth = z);
    T = T.add_variable(name='raw_time',expr=pl.col('time'));
    T = T.use_datetime(start_date = start_time);
    
    # Compute density
    #T.data = (T.data.explode(
    #                columns=[col for col in T.data.columns if T.data.schema[col] == pl.List]
    #                )
    #            .with_columns(
    #                gsw.p_from_z(-pl.col('depth'),pl.col('lat')).alias('p')
    #                )
    #            .with_columns(
    #                gsw.SA_from_SP(pl.col('salt'),pl.col('p'),pl.col('lon'),pl.col('lat')).alias('SA')
    #                )
    #            .with_columns(
    #                gsw.CT_from_t(pl.col('SA'),pl.col('temp'),pl.col('p')).alias('CT')
    #                )
    #            .with_columns(
    #                gsw.sigma0(pl.col('SA'),pl.col('CT')).alias('sigma0')
    #                )
    #            .group_by(by='id', maintain_order=True)
    #            .agg(pl.all())
    #         )
    
    if save:
        save_tracmass(T,project,case,oname)
    
    return T;

def read_tracmass_parallel(project,case,
                  oname,ext='run',
                  inpath = '/home/jdo043/DATA/ArMOC/Tracmass/output/',
                  save=False,n_files = None,start_time = None,w_file = 0,wint=[None,None]):

    if ext == 'run':
        files = sorted(ls(inpath+project+'/'+case+'/split_'+oname+'_*_'+ext+'.csv'))
    else:
        files = sorted(ls(inpath+project+'/'+case+'/*_'+oname+'_'+ext+'.csv'))

    onames = [f.split('_'+ext+'.csv')[0].split('/')[-1] for f in files][0:n_files][wint[0]:wint[1]]
    
    if len(onames) == 1:  
        fname = [f.split('_'+ext+'.csv')[0].split('/')[-1] for f in files][w_file]
        if start_time is None:
            date = fname.split('_')[1]
            s_time = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
        else:
            s_time = start_time
        return read_tracmass(project,case,fname,ext,inpath,save,start_time = s_time)
    
    T_frames = []
    
    # If there are several ini dates, calculate offsets to add
    ini_dates = sorted(set([o.split('_')[1] for o in onames]))
    offset = {}
    t_offset = 0
    for ii in ini_dates:
        #print(ii)
        offset[ii] = t_offset
        tt_ini = read_tracmass(project,case,'output_'+ii,'ini',inpath,start_time = ii[0:4]+'-'+ii[4:6]+'-'+ii[6:8])
        t_offset += len(tt_ini)
        
    #print(offset)
    
    for ona in onames:
        print('Reading',ona,'...')
        date = ona.split('_')[1]
        if start_time is None:
            s_time = date[0:4]+'-'+date[4:6]+'-'+date[6:8]
        else:
            s_time = start_time
        try:
            t_curr = read_tracmass(project,case,ona,ext,inpath,save,start_time = s_time)
            t_curr.data = t_curr.data.with_columns(pl.col('id')+ offset[date])
            T_frames.append(t_curr)
            
        except Exception as e:
            print(e)
            pass
    
    # Concatenate frames
    print('Concatenating...')
    data = pl.concat([t.data for t in T_frames])
    
    return ltt.TrajFrame(source=data,condense=False);
        
def save_tracmass(T,project,case,oname='output',max_len = 100000, inpath = '/home/jdo043/DATA/ArMOC/Tracmass/output/'):
    
    vars = T.data.columns
    vars.remove('id')
    vars.remove('vt')

    n_traj = len(T)
    max_n_obs = min(max([len(T.data[i]['time'][0]) for i in range(n_traj)]),max_len)

    data_vars = {c:(['traj','obs'],np.zeros((n_traj,max_n_obs))*np.nan) for c in vars}
    data_vars['vt'] = (['traj'],np.array([T.data[i]['vt'][0][0] for i in range(n_traj)]).astype(float))
    data_vars['id'] = (['traj'],np.array([T.data[i]['id'][0] for i in range(n_traj)]).astype(float))

    coords = {'traj':np.arange(n_traj),'obs':np.arange(max_n_obs).astype(float)}
    ds = xr.Dataset(data_vars=data_vars,coords=coords)
    for v in vars:
        for i in range(n_traj):
            ds[v][i,:] = np.pad(T.data[i][v][0][0:max_n_obs],(0,max_n_obs-len(T.data[i][v][0][0:max_n_obs])),constant_values=np.nan)

    encoding = {c:{'_FillValue': 9.9e36,'zlib':True} for c in ds.variables}

    ds.to_netcdf(inpath+project+'/'+case+'/'+oname+'_run.nc',encoding=encoding)
  
def namelist(project,case,start_date,oname = 'output',n_step = None,n_seed = 3,seed_box = None, kill_box = None,year_loop = None):
    file = '/home/jdo043/Tracmass/projects/'+project+'/namelist_'+case+'.in'
    nml = nl.read(file)

    y,m,d = start_date.split('-')
    nml['init_start_date']['startday'] = int(d)
    nml['init_start_date']['startmon'] = int(m)
    nml['init_start_date']['startyear'] = int(y)
    
    if n_step is not None:
        nml['init_run_time']['intrun'] = n_step
        
    if year_loop is not None:
        nml['init_run_time']['loopstartyear'] = year_loop[0]
        nml['init_run_time']['loopendyear'] = year_loop[1]
        
    nml['init_seeding']['tst2'] = n_seed
    
    nml['init_write_traj']['outdatafile'] = oname +'_'+ y+m+d
    
    nl.write(nml,file,force=True)
    
def tmux(command):
    system('tmux %s' % command)

def tmux_shell(command):
    tmux('send-keys "%s" "C-m"' % command)

def runtracmass(project,case,rname,path = '/home/jdo043/Tracmass/',oname='output'):
    makefile = path + 'Makefile'
    
    # Modify Makefile for specific project and case
    with fileinput.FileInput(makefile, inplace=True, backup='.bak') as file:
        for l,line in enumerate(file):
            if l == 6:
                print(line.replace(line, 'PROJECT	          = '+project),end='\n')
            elif l == 7:
                print(line.replace(line, 'CASE              = '+case),end='\n')
            else:
                print(line,end='')
                
    # Start tmux session for the run
    tname = 'tracmass_'+project+'_'+case+'_'+rname
    tmux('new-session -d -t '+tname)
    #tmux('tmux select-window -t '+tname)
    
    tmux_shell('cd ~/Tracmass/')
    # Start the run
    tmux_shell('make clean')
    tmux_shell('make')
    tmux_shell('./runtracmass')
    #tmux_shell('sleep 600')
    # Unzip and split into seeding time files
    y,m,d = rname.split('-')
    if project == 'GLORYS12_ORCA':
        project = 'GLORYS12_native'
        
    file = '/home/jdo043/DATA/ArMOC/Tracmass/output/'+project+'/'+case+'/'+oname+'_'+y+m+d+'_run.csv'
    try:
        tmux_shell('gunzip -f '+file+'.gz')
    except:
        pass
    tmux_shell('conda activate lt_tool')
    tmux_shell('python /home/jdo043/scripts/ArMOC/split_tm.py '+project+' '+case+' '+rname)
    tmux_shell('rm -rf '+file)
    tmux_shell('sleep 300')
    tmux_shell('exit')
    
    print('Started run on tmux session ',tname)

def splittracmass(project,case,rname):
    y,m,d = rname.split('-')
    oname = y+m+d
    
    iniDir = '/home/jdo043/DATA/ArMOC/Tracmass/output/'+project+'/'+case+'/'
    iniFile = 'output_'+y+m+d+'_ini.csv'
    
    runDir = '/home/jdo043/DATA/ArMOC/Tracmass/output/'+project+'/'+case+'/'
    runFile = 'output_'+oname+'_run.csv'
    
    outDir = '/home/jdo043/DATA/ArMOC/Tracmass/output/'+project+'/'+case+'/'
   
    outFile = "split_"+oname+"_"
    
    col_names = ['id', 'x', 'y', 'z', 'vt', 't', 'wall', 'temp', 'salt', 'dens']
    col_dtypes = {'id': int,'x': float,'y': float,'z': float,'vt': float,
                  't': float,'wall': int,'temp': float,'salt': float,'dens': float} 
   
    df_ini = pd.read_csv(iniDir+iniFile, names=col_names) #, dtype=col_dtypes)

    tlevels = np.unique(df_ini.t)
    if np.min(tlevels) < 0:
        tlevels = np.flipud(tlevels)

    dates = np.datetime64(rname) + np.timedelta64(1, 's') * tlevels
    date_str = pd.to_datetime(dates).strftime('%Y_%m_%d')

    nsteps = len(tlevels)
    id_min = 0
    id_offset = 0
    ds = (pl.read_csv(runDir+runFile,new_columns=col_names,has_header=False,dtypes=col_dtypes,
                              null_values=['********','****************']))

    for n in tqdm(range(nsteps)):
        id_max = np.max(df_ini[df_ini.t == tlevels[n]].id)
        
        ds_scan = ds.filter((pl.col("id") <= id_max) & (pl.col("id") > id_min))

        id_min = id_max
        df_run = ds_scan.to_pandas()

        df_run.columns = col_names
        df_run.astype(col_dtypes)
        try:
            df_run = df_run.drop(columns=['mask'])
        except:
            pass
  
        fname = outFile + date_str[n] + '_run.csv'
        df_run.to_csv(path_or_buf=outDir+fname, header=False, index=False,compression='gzip')
   
    
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

ibcao = xr.open_dataset('/home/jdo043/DATA/ArMOC/ibcao_8000m.nc')

def plot_testcase(project,case,oname='output',n_part=100,save=False,lat_s = 60,ext='run',ptype='scatter',max_time=100000):
    inpath = '/home/jdo043/DATA/ArMOC/Tracmass/output/'+project+'/'+case+'/'
    T = read_tracmass(project,case,oname,save=save,ext=ext)
    
    prob = T.compute_probability(bin_res = 0.1).summary_data
    display(T)
    
    # Plot heatmap of particle locations
    plt.figure(figsize=(10,10))
    ax = plt.subplot(1,1,1,projection=ccrs.NorthPolarStereo())
    plt.pcolormesh(prob.lon,prob.lat,np.log(prob.probability),
                   transform= ccrs.PlateCarree(),vmax=-9,cmap='inferno')
    plt.colorbar()
    ax.coastlines(resolution='10m',zorder=9)
    ax.gridlines()
    polarCentral_set_latlim([lat_s,90], ax)
    plt.savefig('/home/jdo043/plots/ArMOC/Lagrangian/'+project+'_'+case+'_'+oname+'_density.png',dpi=300)
    ################################
    min_time = 0
    #T.plot_trajectories(sample_size=10)
    plt.figure(figsize=(10,20))

    ax = plt.subplot(2,1,1,projection=ccrs.NorthPolarStereo())

    #ax = plt.subplot(2,1,1,projection=ccrs.PlateCarree())
    #ax.set_extent([-25,0,55,75])
    levels = [100,300,500,750,1000,1500,2000,3000,4000,5000]
    #levels = np.arange(100,5000,300)
    plt.contourf(ibcao.x*1.017,ibcao.y*1.017,-ibcao.z,
                levels = levels,
                cmap='Blues',alpha=0.2)

    ax.coastlines(resolution='10m',zorder=9)
    ax.gridlines()
    # Pick n_part random particles 
    T = T.data.sample(n=min(n_part,len(T)))
    for i in range(len(T)):
        #print(T.data[i]['lon'][0])
        if ptype == 'scatter':
            ss = ax.scatter(T[i]['lon'][0][min_time:max_time],T[i]['lat'][0][min_time:max_time],
                   c = T[i]['depth'][0][min_time:max_time],transform= ccrs.PlateCarree(),
                        cmap='viridis_r',vmin=50,vmax=500,s=0.2,zorder=10)
        elif ptype == 'plot':
            ax.plot(T[i]['lon'][0][min_time:max_time],T[i]['lat'][0][min_time:max_time],
                   transform= ccrs.PlateCarree(),marker= '.',markersize=0.2,color='r',alpha=1 / (np.log(n_part)/4 + 1) ,lw=0.1,zorder=10)
    polarCentral_set_latlim([lat_s,90], ax)

    if ptype == 'scatter':
        plt.colorbar(ss,orientation='vertical',label='Depth [m]',aspect=80)
    

    alpha= min(1/(n_part/10),1)
    plt.subplot(8,1,5)
    for i in range(0,len(T)):
        plt.plot(T[i]['time'][0][min_time:max_time],T[i]['temp'][0][min_time:max_time],color='k',alpha=alpha)
    #plt.xlim(dt.date(1995,1,1),dt.date(1995,5,1))
    plt.ylabel('Temperature [˚C]')
    plt.subplot(8,1,6)
    for i in range(0,len(T)):

        plt.plot(T[i]['time'][0][min_time:max_time],T[i]['depth'][0][min_time:max_time],color='k',alpha=alpha)
    plt.ylabel('Depth [m]')
    plt.ylim(1,750)
    #plt.gca().set_yscale('log')
    #plt.xlim(dt.date(1995,1,1),dt.date(1995,5,1))

    plt.subplot(8,1,7)
    for i in range(0,len(T)):
        plt.plot(T[i]['time'][0][min_time:max_time],T[i]['salt'][0][min_time:max_time],color='k',alpha=alpha)
    plt.ylabel('Salinity ()')
    #plt.xlim(dt.date(1995,1,1),dt.date(1995,5,1))

    plt.subplot(8,1,8)
    for i in range(0,len(T)):
        plt.plot(T[i]['time'][0][min_time:max_time],T[i]['dens'][0][min_time:max_time],color='k',alpha=alpha)
    plt.ylabel('Density (g/kg)')
    #plt.xlim(dt.date(1995,1,1),dt.date(1995,5,1))
    plt.subplots_adjust(hspace=0.15)
    #plt.savefig('/home/jdo043/plots/ArMOC/Lagrangian/Tracmass_test_Arctic_3D.png',dpi=300)