# needs module load R/3.4.1
library(ProFound)
save_path = '/home/rs17612/DPS_Comparison/'
Image_Path = "Sim_imageKIds_combined_2.fits"
image = Rfits_read_image(Image_Path)$imDat

segim_new_expand_tol100=profoundProFound(image,skycut=2, reltol=-10,cliptol=100, pixcut=3,
                                        expandsigma=2,expand=1,roughpedestal = TRUE, tolerance = 15,verbose=TRUE,
                                        rotstats=TRUE,boundstats=TRUE, nearstats=TRUE, groupstats=TRUE, groupby='segim',
                                        deblend=TRUE)

Rfits_write_image(segim_new_expand_tol100$segim, "PF_SIMSegim_seed.fits") # segmentation map of objects
Rfits_write_image(segim_new_expand_tol100$group$groupim, "ProFoundGroupIm_Sims_seed.fits") # group masks
#Rfits_write_image(segim_new_expand_tol100$objects, "ProFoundOut.fits")
write.table(segim_new_expand_tol100$segstats, "ProFound_Catalogue_segstats_Sims_seed.fits") # catalogue of objects
write.table(segim_new_expand_tol100$groupstats, "ProFound_Catalogue_groupstats_Sims_seed.fits") # catalogue of groups
