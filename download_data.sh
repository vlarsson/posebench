#!/bin/sh

mkdir data
cd data
mkdir relative
cd relative

# SfM fisheye image pairs from Terekhov and Larsson (ICCV'23)
wget -N http://vision.maths.lth.se/viktor/posebench/relative/fisheye_grossmunster_4342.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/fisheye_kirchenge_2731.h5

# Image pairs from IMC 2021
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_lincoln_memorial_statue.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_piazza_san_marco.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_london_bridge.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_sagrada_familia.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_british_museum.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_milan_cathedral.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_st_pauls_cathedral.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_florence_cathedral_side.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/imc_mount_rushmore.h5

# Image pairs from ScanNet. Evaluation protocol from SuperGlue paper (Sarlin et al.)
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_spsg.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_sift.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_roma.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_dkm.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_aspanformer.h5

# Image pairs from MegaDepth
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_sift.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_spsg.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_splg.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_roma.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_dkm.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/megadepth1500_aspanformer.h5

cd ..
mkdir absolute
cd absolute

# Queries from 7 Scenes dataset including triangulated lines from LIMAP. (SP&SG + SOLD2 + LSD)
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/7scenes_heads.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/7scenes_stairs.h5

# Queries from Cambridge Landmark (SP&SG + SOLD2 + LSD)
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/cambridge_landmarks_KingsCollege.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/cambridge_landmarks_StMarysChurch.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/cambridge_landmarks_OldHospital.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/cambridge_landmarks_GreatCourt.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/cambridge_landmarks_ShopFacade.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/eth3d_130_dusmanu.h5

# Calibration pattern images collected from BabelCalib
mkdir calib
cd calib
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_single_plane_2012-A0.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_Snapdragon_outdoor_45.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_TUMVI.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_DAVIS_outdoor_forward.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_single_plane_3136-H0.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov01.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_BM2820.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_GOPR.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_Snapdragon_indoor_forward.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov00.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov07.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_Snapdragon_outdoor_forward.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_cube_ov00.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_cube_ov01.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_single_plane_5501-C4.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_GOPRO.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_ENTANIYA.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_Fisheye2.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov06.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_BM4018S118.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_Omni.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_cube_ov03.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_KaidanOmni.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_Snapdragon_indoor_45.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov05.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_Ladybug.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_single_plane_130108MP.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_DAVIS_indoor_45.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_Fisheye1.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_cube_ov02.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OV_corner_ov04.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_DAVIS_indoor_forward.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_BF2M2020S23.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/UZH_DAVIS_outdoor_45.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_EUROC.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_MiniOmni.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/Kalibr_BF5M13720.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_VMRImage.h5
wget -N http://vision.maths.lth.se/viktor/posebench/absolute/calib/OCamCalib_Fisheye190deg.h5

cd ../..

mkdir homography
cd homography
# Homography dataset (validation set) from Barath et al. (CVPR 2023?)
wget -N http://vision.maths.lth.se/viktor/posebench/homography/barath_Alamo.h5
wget -N http://vision.maths.lth.se/viktor/posebench/homography/barath_NYC_Library.h5

cd ../..