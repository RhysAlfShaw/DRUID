
import glob
import os 
import warnings

from regions import Regions
import numpy as np
import pandas as pd

from astropy.table import Table, vstack

warnings.filterwarnings('ignore')


def find_files_in_directory(directory_path, file_name_structure):
 
    file_paths = glob.glob(os.path.join(directory_path, file_name_structure))

    return file_paths

# Example usage:
directory_path = '/data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/'
file_name_structure = 'facet_*_catalogue.fits'  # Use asterisk (*) as a wildcard to match any number
found_files = find_files_in_directory(directory_path, file_name_structure)

print(len(found_files))

# Need Ra and Dec output to improve.


table_list = []
for file in found_files:
    # get facet number 
    facet_number = file.split('/')[-1].split('_')[1]
    print(facet_number)
    table = Table.read(file)
    table['facet_number'] = facet_number
    table_list.append(table)
    
    
full_table = vstack(table_list)

full_table.write('/data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/TRSFfull_lockman_catalogue.fits', overwrite=True)

# do duplicate filtering

DRUIDcatalogue = full_table.to_pandas()

base_PATH_region = '/data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/facet_{:02d}.reg'

region_list = []
facet_numbers = []
for i in range(1,26):
    PATH = base_PATH_region.format(i)
    facet_number = i
    facet_numbers.append(facet_number)
    reg = Regions.read(PATH)
    region_list.append(reg)

# combine the two lists into a 2xN array
region_list = np.array(region_list)
facet_numbers = np.array(facet_numbers)
facet_numbers = facet_numbers.reshape(25,1)

Facet_regions = np.concatenate((facet_numbers, region_list), axis=1)

index = 0
while len(DRUIDcatalogue) > index:
    print(index, len(DRUIDcatalogue))
    radius = 0.15 # in arcsec
    radius_deg = radius/3600
    #for index, row in DRUIDcatalogue.iterrows():
    row = DRUIDcatalogue.iloc[index]    
    index += 1
    CenterRA = row['RA']
    CenterDEC = row['DEC']
    # find the sources within the radius
    matches = None
    # filter DRUIDcatalogue to only include sources within the boundary
    DRUIDcatalogue_filtered = DRUIDcatalogue[(DRUIDcatalogue['RA'] > CenterRA - radius_deg) & (DRUIDcatalogue['RA'] < CenterRA + radius_deg) & (DRUIDcatalogue['DEC'] > CenterDEC - radius_deg) & (DRUIDcatalogue['DEC'] < CenterDEC + radius_deg)]
    if len(DRUIDcatalogue_filtered) == 0:
        #print('no matches found')
        continue
    else:
        for jndex, altrow in DRUIDcatalogue_filtered.iterrows():
            if altrow['RA'] == CenterRA and altrow['DEC'] == CenterDEC:
                continue
            dist = np.sqrt((altrow['RA'] - CenterRA)**2 + (altrow['DEC'] - CenterDEC)**2)
            if dist < radius_deg:
                # create an array of matches
                try:
                    if matches == None:
                        matches = altrow.to_numpy()
                except ValueError:
                    matches = np.vstack((matches, altrow.to_numpy()))
                else:
                    matches = np.vstack((matches, altrow.to_numpy()))
                    
        if matches is None:
            #print('no matches found')
            continue
            
        else:
            matches = np.vstack((matches, row.to_numpy())) # add the original row to the matches
        
            Distances = []
            for Index, Element in enumerate(matches):
                
                Facet_number = int(Element[28]) # the facet number is stored as a byte string
                facet_center = Facet_regions[Facet_number-1,1].center
                facet_center_ra = facet_center.ra.deg
                facet_center_dec = facet_center.dec.deg
                dist = np.sqrt((Element[26] - facet_center_ra)**2 + (Element[27] - facet_center_dec)**2)    
                Distances.append(dist)
                
            min_dis_index = np.argmin(Distances)
            min_dis_row = matches[min_dis_index]
            #print(min_dis_row)

        # remove matches from the dataframe and add the closest match back in.

        # remove the matches
            for Index, Element in enumerate(matches):
                DRUIDcatalogue = DRUIDcatalogue.drop(DRUIDcatalogue[(DRUIDcatalogue['RA'] == Element[26]) & (DRUIDcatalogue['DEC'] == Element[27])].index)

        # add the closest match back in
            DRUIDcatalogue = DRUIDcatalogue.append(pd.Series(min_dis_row, index=DRUIDcatalogue.columns), ignore_index=True)
    

# save DRUIDcatalogue to fits file.
DRUIDcatalogueClass0andClass2 = DRUIDcatalogue[(DRUIDcatalogue['Class'] == 0) | (DRUIDcatalogue['Class'] == 2) | (DRUIDcatalogue['Class'] == 4) | (DRUIDcatalogue['Class'] == 1)]

DRUIDcatalogue = Table.from_pandas(DRUIDcatalogue)
DRUIDcatalogueClass0andClass2 = Table.from_pandas(DRUIDcatalogueClass0andClass2)
DRUIDcatalogue.write('/data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/TRSFfull_lockman_catalogue_filtered.fits', overwrite=True)
DRUIDcatalogueClass0andClass2.write('/data/typhon2/Rhys/data/LoFAR/Lockman_hole/facets/TRSFfull_lockman_catalogue_filteredClass0andClass2.fits', overwrite=True)