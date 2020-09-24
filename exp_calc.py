import os
import numpy as np
import pandas as pd

from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.units import kpc,arcsec

from tqdm import tqdm
from yaml import load

#Check if Cparser is available
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class Utils():

    @staticmethod
    def XMM_possition_to_pixel(
                                xmm_pos:tuple,
                                object_pos:tuple,
                                *,
                                pix_to_deg:float=1.208e-3,
                                center:tuple=(256.5,256.5),
                                origin:str="upper") -> tuple:
        ra_xmm,dec_xmm = xmm_pos
        ra,dec = object_pos
        theta = np.cos(np.pi/360*(dec+dec_xmm))
        delta_ra = -(ra_xmm -ra)*theta
        delta_dec = dec_xmm - dec
        x_c,y_c = center
        deltx = delta_ra/pix_to_deg
        delty = delta_dec/pix_to_deg
        out_x = round(x_c - deltx,0)
        out_y = round(y_c - delty,0)
        if origin=="upper":
            out_y = 512-out_y
        elif origin=='lower':
            pass
        return out_x,out_y

    @staticmethod
    def XMM_pixel_to_position(
                                pixel_coords:tuple,
                                obs_center:tuple,
                                *,
                                pix_to_deg:float=1.208e-3,
                                center:tuple =(256.5,256.5)) -> tuple:
        x,y = pixel_coords
        ra_c,dec_c = obs_center
        x_c,y_c = center
        delx = x_c -x
        dely = y_c -y
        theta = np.cos(dec_c * np.pi/180)
        del_RA = -delx*pix_to_deg/theta
        del_DEC = dely*pix_to_deg
        return ra_c - del_RA, dec_c - del_DEC

    @staticmethod
    def cmask(index:int,radius:float,array_size:tuple):
        b,a = index
        nx,ny = array_size
        y,x = np.ogrid[-a:nx-a,-b:ny-b]
        mask = x*x + y*y <= radius*radius
        return mask

    @staticmethod
    def obsid_fix(x):
        return str(x).zfill(10)

class EEcalculator():
    """
        Simple parent class that can calcualte the effective exposure for a
        given cluster or redmapper positon.
    """

    def __init__(self):

        self._config_file = "exp_calc.yaml"
        self._params = self._load_config()


    def _set_archive_path(self, path:str):
        """
            Boring Place holder
        """
        if os.path.isdir(path):
            self._archive_path = path
        else:
            print("Directory does not exist", path)

    def _get_archive_path(self):
        return self._archive_path

    archive_path = property(_get_archive_path, _set_archive_path)

    def _get_config_file(self):
        """
            Boring Place holder
        """
        return self._config

    def _set_config_file(self, filename:str):
        """
            Boring Place holder
        """
        if os.path.isfile(filename):
            self._config = filename
            self._params = self._load_config()
        else:
            print("File not found: ", filename)

    config_file = property(_get_config_file,_set_config_file)

    def _load_config(self) -> dict:
        with open(self._config_file, 'r') as fin:
            return load(fin, Loader=Loader)


    def calc_percentile_exposure(self,
                *,
                obsid: str,
                xmm_pos:tuple,
                object_pos:tuple,
                redshift:float,
                r_lambda:float,
                cosmo)              -> float:

        index = Utils.XMM_possition_to_pixel(   xmm_pos,
                                                object_pos,
                                                origin='lower'
                                            )

        a=cosmo.arcsec_per_kpc_proper(redshift)
        r=0.8*r_lambda*1000

        pix_rad=(r*a/4.35)*(1.*kpc/arcsec)
        pix_rad=round(pix_rad.value)
        mask = Utils.cmask(index,pix_rad,(512,512))
        lower_value = 0
        percentile = self._params["percentile_level"]
        for cam in self._params["expmap_images"].keys():
            PATH = os.path.join(self._params["archive_path"],
                                self._params["images_path"],
                                self._params["expmap_images"][cam])
            PATH=PATH.format(obsid=obsid)
            if os.path.isfile(PATH):
                try:
                    data = fits.open(PATH)[0].data
                    data = data[mask]
                    data_sequential = list(sorted(data.flatten().tolist()))
                    percentile_position = round(len(data_sequential)*percentile/100)
                    if cam=="PN":
                        lower_value+=data_sequential[percentile_position]
                    else:
                        lower_value+=data_sequential[percentile_position]/2.0
                except (IOError,IndexError):
                    continue
        return lower_value

    def calc_effective_exposure(self,
                        *,
                        obsid:str,
                        xmm_pos: tuple,
                        object_pos: tuple,
                        redshift:float,
                        r_lambda:float,
                        cosmo,
                        worst_case:bool=False):
            index = Utils.XMM_possition_to_pixel(    xmm_pos,
                                            object_pos,
                                            origin='lower')
            if worst_case:
                a=cosmo.arcsec_per_kpc_proper(redshift)
                r=0.8*r_lambda*1000
                pix_rad=(r*a/4.35)*(1.*kpc/arcsec)
                del_X = index[0] -256.5
                del_Y = index[1] -256.5
                del_R = del_X**2+del_Y**2
                ratio = np.sqrt((pix_rad**2)/del_R)
                worst_x = int(index[0] + del_X*ratio)
                worst_y = int(index[1] + del_Y*ratio)
                index = (worst_x,worst_y)

            mask = Utils.cmask(index,self._params["effective_radius"],(512,512))
            mean = 0
            median = 0
            for cam in self._params["expmap_images"].keys():
                PATH = os.path.join(self._params["archive_path"],
                                    self._params["images_path"],
                                    self._params["expmap_images"][cam])
                PATH=PATH.format(obsid=obsid)
                if os.path.isfile(PATH):
                    try:
                        data = fits.open(PATH)[0].data
                        data = data[mask]
                        if not data.size:
                            continue
                        if cam=="PN":
                            median+=np.median(data)
                            mean+=np.mean(data)
                        else:
                            median += np.median(data)/2
                            mean += np.mean(data)/2
                    except IOError:
                        continue
            return mean,median

    def _get_cosmology(self):
        if "FlatLambdaCDM" in self._params["cosmology"].keys():
            args = self._params["cosmology"]["FlatLambdaCDM"]
            if "args" in args.keys():
                cosmo = FlatLambdaCDM(H0=args["H0"], Om0=args["Om0"], *args["args"])
            else:
                cosmo = FlatLambdaCDM(H0=args["H0"], Om0=args["Om0"])
            return cosmo
        else:
            raise ImportError("Cosmology is not set properly in config file")

    def get_effective_exposures(self, infile:str, index_col:int, outfile:str=""):
        df = pd.read_csv(infile,index_col=index_col)

        infile_keys = self._params["infile_keys"]
        df[infile_keys["obsid"]] = df[infile_keys["obsid"]].apply(Utils.obsid_fix)

        df_obs = pd.read_csv(self._params["XMM_obs"], index_col=None)
        df_obs[infile_keys["obsid"]] = df_obs[infile_keys["obsid"]].apply(Utils.obsid_fix)


        print("Beginning Effective exposure calculation for csv:", infile)
        out_dict ={}
        row=-1
        for idx in tqdm(df.index):
            row+=1
            RA = df.loc[idx, infile_keys["ra"]]
            DEC = df.loc[idx, infile_keys["dec"]]
            _id = df.loc[idx, infile_keys["_id"]]
            r_lam = df.loc[idx, infile_keys["r_lam"]]
            redshift = df.loc[idx, infile_keys["redshift"]]
            obsid = df.loc[idx, infile_keys["obsid"]]
            tmp = df_obs[df_obs["OBSID"]==obsid]
            RA_xmm = tmp.RA.values[0]
            DEC_xmm = tmp.DEC.values[0]
            cosmo = self._get_cosmology()
            xmm_pos = (RA_xmm,DEC_xmm)
            object_pos = (RA,DEC)
            mean,median = self.calc_effective_exposure(
                                    obsid=obsid,
                                    xmm_pos=xmm_pos,
                                    object_pos=object_pos,
                                    redshift=redshift,
                                    r_lambda=r_lam,
                                    cosmo=cosmo,
                                    worst_case=False)

            worst_mean,worst_median = self.calc_effective_exposure(
                                    obsid=obsid,
                                    xmm_pos=xmm_pos,
                                    object_pos=object_pos,
                                    redshift=redshift,
                                    r_lambda=r_lam,
                                    cosmo=cosmo,
                                    worst_case=True)
            lower_bound = self.calc_percentile_exposure(
                                    obsid=obsid,
                                    xmm_pos=xmm_pos,
                                    object_pos=object_pos,
                                    redshift=redshift,
                                    r_lambda=r_lam,
                                    cosmo=cosmo)

            out_dict[row] = {"RM_ID":_id,
                            "OBSID": obsid,
                            "MEAN": mean,
                            "MEDIAN": median,
                            "WORST_MEAN": worst_mean,
                            "WORST_MEDIAN": worst_median,
                            "PERCENTILE": lower_bound
                            }

        if outfile:
            pd.DataFrame.from_dict(out_dict).to_csv(outfile)
        else:
            for row in out_dict.keys():
                for key,value in out_dict[row].items():
                    print(key,value)
                print("\n")


if __name__ == "__main__":
    E = EEcalculator()
    E.get_effective_exposures(infile="example.csv",index_col=None)
