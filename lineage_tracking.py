
import numpy as np
import matplotlib.pyplot as plt

def index_dataframe_by_frame(label_df, frame):
   return label_df[label_df.frame==frame]

def index_dataframe_by_distance(label_df, max_distance):
   return label_df[label_df.distance<=max_distance]

def index_dataframe_by_area(label_df, cell_area, area_range):
   return label_df[label_df.area.between(area_range[0]*cell_area, area_range[1]*cell_area)]

def get_minimum_distance_dataframe(label_df):
   minimum_distance = label_df.distance.min()
   return label_df[label_df.distance==minimum_distance]

def remove_keys_with_common_values(dictionary, duplicated_values_list):
   dictionary = {k:v for k,v in dictionary.items() if v not in duplicated_values_list}
   return dictionary


def linking_cell_labels(label_df, max_radius, area_range):
   
   n_frames = label_df.frame.max()
   print(n_frames, 'frames in total')
   
   label_df['x'] = label_df['centroid-1']
   label_df['y'] = label_df['centroid-0']
   
   linkage_dictionary = {}
   used_ids = []
   duplicated_ids = []
   
   for idx, row in label_df.iterrows():
      
      frame = row.frame
      current_x = row.x
      current_y = row.y
      cell_area = row.area
      cell_id = row.cell_id
      next_frame = frame+1
      
      next_labels_df = index_dataframe_by_frame(label_df, next_frame)
      next_labels_df['x_diff'] = next_labels_df['x'] - current_x
      next_labels_df['y_diff'] = next_labels_df['y'] - current_y
      next_labels_df['distance'] = np.sqrt(next_labels_df['x_diff']**2 + next_labels_df['y_diff']**2)
      
      linked_df = next_labels_df[(next_labels_df.distance<=max_radius)&(next_labels_df.area.between(area_range[0]*cell_area, area_range[1]*cell_area))]
      
      if linked_df.shape[0] == 1:
         linked_id = linked_df.cell_id.values[0]
      elif linked_df.shape[0]>1:
         linked_df = get_minimum_distance_dataframe(linked_df)
         linked_id = linked_df.cell_id.values[0]
      else:
         linked_id = 'none'
      
      if linked_id != 'none': 
         linkage_dictionary[cell_id] = linked_id
         if linked_id in used_ids:
            duplicated_ids.append(linked_id)
            print(f"cell {linked_id} has already been linked to another cell...")
         else:
            used_ids.append(linked_id)
            print(f"cell {cell_id} in frame {frame} linked to cell {linked_id} to frame {next_frame}")
      else:
         print(f"cell {cell_id} in frame {frame} not linked to a cell")
   
   return linkage_dictionary, duplicated_ids


def link_lineages(cell_id_list, all_cells_list, linkage_dictionary):
   if cell_id_list[-1] in linkage_dictionary:
      
      linked_index = linkage_dictionary[cell_id_list[-1]]        
      cell_id_list.append(linked_index)
      all_cells_list.append(linked_index)
      return link_lineages(cell_id_list, all_cells_list, linkage_dictionary)
   else:
      return cell_id_list, all_cells_list


def get_cell_lineages(linkage_dictionary, duplicated_values_list):

   non_duplex_dict = remove_keys_with_common_values(linkage_dictionary, duplicated_values_list)
   
   i = 0
   cell_traj_dict = {}
   all_cells_list = []
   
   for seed in non_duplex_dict:
      if seed not in all_cells_list:
         all_cells_list.append(seed)
         cell_id_list = [seed]
         cell_id_list, all_cells_list = link_lineages(cell_id_list, all_cells_list, non_duplex_dict)
         for cid in cell_id_list:
               traj_string = 'cell_traj_'+str(i)
               cell_traj_dict[cid] = traj_string
         print(f"trajectory ID {traj_string} with {len(cell_id_list)} positions: {cell_id_list}")
         i+=1
   return cell_traj_dict


def tracking_lineages(label_df, max_radius=10, area_range=(0.95, 1.15)):
   linkage_dict, duplicated_indexes = linking_cell_labels(label_df, max_radius, area_range)
   cell_traj_dict = get_cell_lineages(linkage_dict, duplicated_indexes)
   label_df['traj_id'] = label_df.cell_id.map(cell_traj_dict)
   return label_df

def show_cell_trajectory(cell_df, cropped_masks_dict):
   
   cell_df = cell_df.sort_values('frame')
   max_cell_id = cell_df[cell_df.area==cell_df.area.max()].cell_id.values[0]
   max_mask = cropped_masks_dict[max_cell_id]
   max_height = max_mask.shape[0]+10
   max_width = max_mask.shape[1] + 10
   
   i = 0
   for idx, row in cell_df.iterrows():
      
      cell_mask = cropped_masks_dict[row.cell_id]
      added_array = np.zeros((max_height-cell_mask.shape[0],cell_mask.shape[1]), dtype=float) 
      new_image = np.concatenate((cell_mask, added_array), axis=0) 
      added_array = np.zeros((new_image.shape[0],max_width-new_image.shape[1]), dtype=float) 
      new_image = np.concatenate((new_image, added_array), axis=1) 
            
      if i == 0:
         original_image = new_image
         if new_image.shape[0] > new_image.shape[1]:
            image_string = 'narrow'
         else:
            image_string = 'long'
      elif i > 0:
         if image_string == 'narrow':
            original_image = np.concatenate((original_image, new_image), axis=1)
         elif image_string == 'long':
            original_image = np.concatenate((original_image, new_image), axis=0)
      i+=1
   
   plt.imshow(original_image, cmap='gray')
   plt.xticks([])
   plt.yticks([])
   plt.show()
