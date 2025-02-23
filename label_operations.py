import time
import os
import re
from skimage import io
from skimage.measure import regionprops_table
import pandas as pd
import matplotlib.pyplot as plt


def get_cell_mask_files(cell_labels_path, file_suffix):
   directory_list = os.listdir(cell_labels_path)
   return [f for f in directory_list if file_suffix in f]


def get_time_xy_channel_name(file_name, iteration_axis='xytc'):
   if iteration_axis == 'xytc':
      pattern = r'(xy)(\d+)(t)(\d+)(c)(\d+)'
      match = re.findall(pattern, file_name)[0]
      # print(match)
      xyp = int(match[1])
      tme = int(match[3])
      cnl = int(match[5])
      return xyp, tme, cnl
   else:
      print(f"iteration axis {iteration_axis} has not been implemented yet")
      

def read_image(image_path):
   return io.imread(image_path)
   

class cell_label_operations(object):
   
   def __init__(self, cell_labels_path, label_suffix='_mask'):
      start = time.time()
        
      labeled_files = get_cell_mask_files(cell_labels_path, label_suffix)
      
      cell_masks = {}
      i = 0
      for fl in labeled_files:
         xyp, tme, cnl = get_time_xy_channel_name(fl)
         cell_labels = read_image(cell_labels_path+'/'+fl)
         cell_masks[tme] = cell_labels
         region_properties = regionprops_table(cell_labels,
                                          properties=('centroid', 'orientation', 'axis_major_length', 'axis_minor_length', 'label', 'area', 'bbox', 'centroid_local'))
         region_table = pd.DataFrame(region_properties)
         region_table['xy_position'] = xyp
         region_table['frame'] = tme
         region_table['cell_id'] = 'xy'+region_table['xy_position'].astype(str)+'_fr'+region_table['frame'].astype(str)+'_lbl'+region_table['label'].astype(str)
         if i == 0:
            final_table = region_table
            i +=1 
         else:
            final_table = pd.concat([final_table, region_table])
      
      final_table['crop-0'] = final_table['bbox-0']-4
      final_table['crop-1'] = final_table['bbox-1']-4
      final_table['crop-2'] = final_table['bbox-2']+4
      final_table['crop-3'] = final_table['bbox-3']+4
      
      self.sensor = cell_labels.shape
      self.cell_masks = cell_masks
      self.xy_position = xyp
      self.region_table = final_table
   
   
   def get_cell_masks(self):
      return self.cell_masks
   

   def remove_cell_out_of_boundaries(self):
      y_max = self.sensor[0]
      x_max = self.sensor[1]

      cell_id_df = self.region_table
      
      wb_df = cell_id_df[(cell_id_df['crop-0'] > 0)&
                           (cell_id_df['crop-1'] > 0)&
                           (cell_id_df['crop-2'] < y_max)&
                           (cell_id_df['crop-3'] < x_max)
                           ]
      
      wb_cells = list(wb_df.cell_id.unique())
      print(f"{len(wb_cells)} cells within bounds")
      return wb_df
   
   
   def crop_masks(self, verbose):
      
      cropped_masks = {}
      
      masks_dict = self.get_cell_masks()
      region_df = self.remove_cell_out_of_boundaries()
      
      for idx, row in region_df.iterrows():
         min_y = row['crop-0']
         min_x = row['crop-1']
         max_y = row['crop-2']
         max_x = row['crop-3']
         if verbose:
            print(f"cropping cell mask with ID {row.cell_id}")
         cropped_masks[row.cell_id] = masks_dict[row.frame][min_y:max_y, min_x:max_x]==row.label

      self.cropped_masks = cropped_masks


   def get_cropped_masks(self):
      return self.cropped_masks
