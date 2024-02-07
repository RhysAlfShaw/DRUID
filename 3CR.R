# R code used to analye 3CR source with ProFound.
# Author: Rhys Shaw
# Date: 16-01-2024s
# Load ProFound

library(ProFound)

Base_PATH = '/data/typhon2/Rhys/data/3CR_ALL_Three/'

PATH_3C401 = paste0(Base_PATH, '3C401')
PATH_3C295 = paste0(Base_PATH, '3C295.fits')
PATH_3C438 = paste0(Base_PATH, '3C438')
PATH_3C452 = paste0(Base_PATH, '3C452')
PATH_3C314P1 = paste0(Base_PATH, '3C314P1')
PATH_3C76P1 = paste0(Base_PATH, '3C76P1')

PATHS = c(PATH_3C401, PATH_3C295, PATH_3C438, PATH_3C452, PATH_3C314P1, PATH_3C76P1)

runProFound = function(image_matrix,path){

    # Run ProFound with the following settings
    
    Results = profoundProFound(image_matrix,
            plot=TRUE,verbose=TRUE,skycut=5,
            rotstats = TRUE,
            boundstats = TRUE,
            nearstats = TRUE, 
            groupstats = TRUE,  
            smooth = FALSE,
            roughpedestal = TRUE,
            tolerance=15,
            cliptol=100,
            expandsigma=2)
    
    # Save the results 
    Rfits_write_image(Results$segim, paste0(path, '_segimPF.fits'))
    Rfits_write_image(Results$group$groupim, paste0(path, '_groupimPF.fits'))
    # save the group stats
    write.table(Results$groupstats, paste0(path, '_groupstatsPF.txt'))
}

# loop though all the PATHS
for (i in 1:length(PATHS)){
    image = Rfits_read_image(PATHS[i],header=TRUE)$imDat
    image_matrix <- matrix(image,nrow=nrow(image),ncol=ncol(image))
    runProFound(image_matrix,PATHS[i])
}