import nd2_to_array as ndtwo
import lineage_tracking as lt # type: ignore
import label_operations as lblo
import os
import background_correction as bkg # type: ignore
import matplotlib.pyplot as plt
import Biviriate_medial_axis_estimation as medax
import numpy as np


def read_ndtwo_images(ndtwo_path):
   return ndtwo.nd2_to_array(ndtwo_path)

def load_images(image_path, fluor_interval, label_suffix):
   
   images_dict = {}
   images_list = os.listdir(image_path)
   images_list = [im for im in images_list if label_suffix not in im]
   
   for im in images_list:
      xyp, tme, cnl = lblo.get_time_xy_channel_name(im, iteration_axis='xytc')
      if tme%fluor_interval==0:
         images_dict[tme] = lblo.read_image(image_path+'/'+im)
   return images_dict


def get_fluorescent_channels(channels):
   
   if 'Phase' in channels:
      phase_channel = 'Phase'
   elif 'Trans' in channels:
      phase_channel = 'Trans'
   elif 'Brightfield' in channels:
      phase_channel = 'Brightfield'
   
   fluor_channels = channels.copy()
   fluor_channels.remove(phase_channel)
   
   print(f"Phase contrast channel: {phase_channel}, fluorescence channels: {fluor_channels}")
   
   return phase_channel, fluor_channels


def subtract_background(image_arrays, phase_channel, fluor_channels):
   
   bkg_cor_images = {}
   
   for ch in fluor_channels:
      bkg_cor_images[ch] = {}
      for tm in image_arrays[ch]:
         phase_image = image_arrays[phase_channel][tm]
         phase_mask = bkg.get_otsu_mask(phase_image)
         signal_image = image_arrays[ch][tm]
         
         bkg_cor_images[ch][tm] = bkg.back_sub(signal_image, phase_mask, dilation=35, estimation_step=128, smoothing_sigma=50, show=False)[0]
   
   return bkg_cor_images
   

class fluorescence_analysis(lblo.cell_label_operations):
   
   def __init__(self, cell_labels_path, label_suffix, images_paths, fluor_interval):
      super().__init__(cell_labels_path, label_suffix)
      
      image_arrays = {}
      
      self.fluor_interval = fluor_interval   
      print('Cropping cell masks within bounds...')
      self.crop_masks(False)
      print('Tracking cell lineages...')
      self.track_df = lt.tracking_lineages(self.remove_cell_out_of_boundaries(), max_radius=10, area_range=(0.975, 1.15))
      print('Loading cell images...')
      for ch in images_paths:
         image_arrays[ch] = load_images(images_paths[ch], fluor_interval, label_suffix)
      
      self.image_arrays = image_arrays
      self.channels = list(image_arrays.keys())
      print('Subtracting fluorescence background...')
      phase_channel, fluor_channels = get_fluorescent_channels(self.channels)
      bkg_cor_images = subtract_background(self.image_arrays, phase_channel, fluor_channels)

      self.phase_channel = phase_channel
      self.fluor_channels = fluor_channels
      self.bkg_cor_images = bkg_cor_images
      

   def get_cell_images(self):
      return self.image_arrays
   
   def get_bkg_corrected_images(self):
      return self.bkg_cor_images
   
   def get_phase_channel(self):
      return self.phase_channel
   
   def get_fluorescence_channels(self):
      return self.fluor_channels
   
   def get_tracked_labels(self):
      return self.track_df
   
   def get_long_trajectories(self, min_length=10):
      label_df = self.get_tracked_labels()
      length_dict = label_df.groupby('traj_id').cell_id.count().to_dict()
      label_df['trajectory_length'] = label_df.traj_id.map(length_dict)
      long_trajectories = label_df[label_df.trajectory_length>=min_length].traj_id.unique().tolist()
      print(f"{len(long_trajectories)} long trajectories with more than {min_length-1} frames")
      return long_trajectories
   
   def get_trajectory_mother(self, trajectory_id):
      track_df = self.get_tracked_labels()
      traj_df = track_df[track_df.traj_id==trajectory_id]
      traj_df = traj_df.sort_values('frame')
      return traj_df.cell_id.values[0]
   

def get_oned_intensity(crop_pad, oned_df, cropped_mask, fluor_image, channel):
    
    min_x, min_y, max_x, max_y = crop_pad
    cropped_fluor_image = fluor_image[min_y:max_y, min_x:max_x]
    oned_df['fluor_'+channel] = cropped_fluor_image[np.nonzero(cropped_mask)]
    
    return oned_df
 
 
def get_oned_fluorescence_stats(track_df, bkg_cor_images, cropped_masks, fluor_channels, long_trajectories):
   
   oned_coords_dict = {}
   medial_axis_dict = {}
   
   for idx, row in track_df.iterrows():
      if row.frame in bkg_cor_images[fluor_channels[0]] and row.traj_id in long_trajectories:
         cropped_cell_mask = cropped_masks[row.cell_id]
         medial_axis_df, cropped_centroid = medax.get_medial_axis(cropped_cell_mask, radius_px=8, half_angle=22, cap_knot=13, max_degree=60, verbose=False)
         oned_df = medax.get_oned_coordinates(cropped_cell_mask, medial_axis_df)
         
         min_y = row['crop-0']
         min_x = row['crop-1']
         max_y = row['crop-2']
         max_x = row['crop-3']
         crop_pad = (min_x, min_y, max_x, max_y)
         
         for ch in fluor_channels:
            oned_df = get_oned_intensity(crop_pad, oned_df, cropped_cell_mask, bkg_cor_images[ch][row.frame], ch)
         
         oned_coords_dict[row.cell_id] = oned_df
         medial_axis_dict[row.cell_id] = medial_axis_df
   
   return medial_axis_dict, oned_coords_dict
      

def get_pole_coords(oned_df, end):
   if end == 1:
      pole_df = oned_df[oned_df.scaled_length == oned_df.scaled_length.max()]
   elif end == -1:
      pole_df = oned_df[oned_df.scaled_length == oned_df.scaled_length.min()]
   x = pole_df.x.values[0]
   y = pole_df.y.values[0]
   return x,y


def calculate_euc_distance(x1, y1, x2, y2):
   return np.sqrt((x2-x1)**2+(y2-y1)**2)


def get_trajectory_polarity(trajectory_id, track_df, oned_coords_dict):
   
   traj_df = track_df[track_df.traj_id==trajectory_id].sort_values('frame')
   traj_df = traj_df[traj_df.cell_id.isin(oned_coords_dict)]
   cell_ids = traj_df.cell_id.values.tolist()
   
   polarity_dict = {}
   
   i = 0
   for cid in cell_ids:
      oned_df = oned_coords_dict[cid]
      if i == 0:
         ref_x, ref_y = get_pole_coords(oned_df, 1)
         polarity_dict[cid] = 1
      elif i > 0:
         pos_x, pos_y = get_pole_coords(oned_df, 1)
         neg_x, neg_y = get_pole_coords(oned_df, -1)
         
         pos_dist = calculate_euc_distance(ref_x, ref_y, pos_x, pos_y)
         neg_dist = calculate_euc_distance(ref_x, ref_y, neg_x, neg_y)
         
         if pos_dist < neg_dist:
            polarity = 1
         elif pos_dist > neg_dist:
            polarity = -1
         elif pos_dist == neg_dist:
            polarity = np.nan
         
         polarity_dict[cid] = polarity
         
      i += 1
      
   return polarity_dict
            

         
   
   
   
         
      
      

      
      
      
      
      